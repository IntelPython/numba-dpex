#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>

#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarrayobject.h>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include "numba/_arraystruct.h"
#include "numba/_numba_common.h"
#include "numba/_pymodule.h"
#include "numba/core/runtime/nrt.h"

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

namespace onemkl = oneapi::mkl;

// forward declarations

extern "C"
{
    // static void DPEX_LAPACK_eigh(arystruct_t *arystruct);
    static void DPEX_LAPACK_eigh(void *b);
}

// static void DPEX_LAPACK_eigh(arystruct_t *arystruct)
static void DPEX_LAPACK_eigh(void *b)
{
    // std::cout << "arystruct->nitems = " << (int)(arystruct->nitems)
    //           << std::endl;
    // std::cout << "arystruct->itemsize = " << (int)(arystruct->itemsize)
    //           << std::endl;

    for (auto platform : sycl::platform::get_platforms()) {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>() << ": ";

        for (auto device : platform.get_devices()) {
            std::cout << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }

    sycl::queue queue(sycl::default_selector_v);
    std::cout << "Default device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    std::vector<float> a = {0.37, 0.91, 0.53, 0.23};
    std::vector<float> w = {0, 0};
    std::vector<float> v = {0, 0, 0, 0};
    std::cout << "a.size() = " << a.size() << ", w.size() = " << w.size()
              << ", v.size() = " << v.size() << std::endl;

    float a_arr[a.size()], w_arr[w.size()], v_arr[v.size()];
    std::copy(a.begin(), a.end(), a_arr);
    std::copy(w.begin(), w.end(), w_arr);
    std::copy(v.begin(), v.end(), v_arr);

    const onemkl::job jobz = oneapi::mkl::job::V;
    const onemkl::uplo upper_lower = oneapi::mkl::uplo::U;
    const std::int64_t n = a.size();
    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t scratchpad_size =
        onemkl::lapack::syevd_scratchpad_size<float>(queue, jobz, upper_lower,
                                                     n, lda);
    float *scratchpad = nullptr;
    std::stringstream error_msg;
    std::int64_t info = 0;
    sycl::event syevd_event;

    std::cout << "w_arr[0] = " << w_arr[0] << std::endl;

    try {
        std::cout << "here.1" << std::endl;
        scratchpad = sycl::malloc_device<float>(scratchpad_size, queue);
        std::cout << "here.2" << std::endl;
        syevd_event = onemkl::lapack::syevd(
            queue,
            jobz, // 'jobz == job::vec' means eigenvalues and eigenvectors are
                  // computed.
            upper_lower, // 'upper_lower == job::upper' means the upper
                         // triangular part of A, or the lower triangular
                         // otherwise
            n,           // The order of the matrix A (0 <= n)
            a_arr, // Pointer to A, size (lda, *), where the 2nd dimension,
                   // must be at least max(1, n) If 'jobz == job::vec', then
                   // on exit it will contain the eigenvectors of A
            lda,   // The leading dimension of a, must be at least max(1, n)
            w_arr, // Pointer to array of size at least n, it will contain
                   // the eigenvalues of A in ascending order
            scratchpad, // Pointer to scratchpad memory to be used by MKL
                        // routine for storing intermediate results
            scratchpad_size, {});
        std::cout << "here.3" << std::endl;
    } catch (onemkl::lapack::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during syevd() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
        info = e.info();
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during syevd() call:\n"
                  << e.what();
        info = -1;
    } catch (...) {
        std::cout << "Something went wrong" << std::endl;
    }
    std::cout << "here.4" << std::endl;
    if (info != 0) // an unexected error occurs
    {
        if (scratchpad != nullptr) {
            sycl::free(scratchpad, queue);
        }
        throw std::runtime_error(error_msg.str());
    }
    std::cout << "here.5" << std::endl;
    syevd_event.wait();

    // sycl::event clean_up_event = queue.submit([&](sycl::handler &cgh) {
    //     cgh.depends_on(syevd_event);
    //     auto ctx = queue.get_context();
    //     cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    // });
    std::cout << "here.6" << std::endl;
    // host_task_events.push_back(clean_up_event);
    return;
}

/*----------------------------------------------------------------------------*/
/*--------------------- The _dpexrt_python Python extension module  -- -------*/
/*----------------------------------------------------------------------------*/

static PyObject *build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value)                                              \
    do {                                                                       \
        PyObject *o = PyLong_FromVoidPtr(value);                               \
        if (o == NULL)                                                         \
            goto error;                                                        \
        if (PyDict_SetItemString(dct, name, o)) {                              \
            Py_DECREF(o);                                                      \
            goto error;                                                        \
        }                                                                      \
        Py_DECREF(o);                                                          \
    } while (0)

    _declpointer("DPEX_LAPACK_eigh", (void *)(&DPEX_LAPACK_eigh));

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

/*--------- Builder for the _dpexrt_python Python extension module  -- -------*/

MOD_INIT(_dpex_lapack_iface)
{
    PyObject *m = NULL;
    PyObject *dpnp_array_type = NULL;
    PyObject *dpnp_array_mod = NULL;

    MOD_DEF(m, "_dpex_lapack_iface", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();
    import_dpctl();

    dpnp_array_mod = PyImport_ImportModule("dpnp.dpnp_array");
    if (!dpnp_array_mod) {
        Py_DECREF(m);
        return MOD_ERROR_VAL;
    }
    dpnp_array_type = PyObject_GetAttrString(dpnp_array_mod, "dpnp_array");
    if (!PyType_Check(dpnp_array_type)) {
        Py_DECREF(m);
        Py_DECREF(dpnp_array_mod);
        Py_XDECREF(dpnp_array_type);
        return MOD_ERROR_VAL;
    }
    PyModule_AddObject(m, "dpnp_array_type", dpnp_array_type);
    Py_DECREF(dpnp_array_mod);

    PyModule_AddObject(m, "DPEX_LAPACK_eigh",
                       PyLong_FromVoidPtr((void *)(&DPEX_LAPACK_eigh)));
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
