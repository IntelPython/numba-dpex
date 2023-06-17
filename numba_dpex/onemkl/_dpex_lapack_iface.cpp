// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <iomanip>
#include <iostream>
#include <sstream>

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
    static void DPEX_LAPACK_eigh(arystruct_t *arystruct);
    // static void DPEX_LAPACK_eigh();
}

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &v)
{
    auto n = v.size();
    os << "[";
    for (auto i = 0; i < n - 1; i++)
        os << v[i] << ", ";
    os << v[n - 1] << "]";
    return os;
}

template <typename T>
std::string format(std::vector<T> &a, int m, int n, int lda)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            ss << " " << std::setw(6) << a[i * lda + j];
        if (i < m - 1)
            ss << '\n';
    }
    return ss.str();
}

void list_platforms()
{
    for (auto platform : sycl::platform::get_platforms()) {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>() << ": ";

        for (auto device : platform.get_devices())
            std::cout << device.get_info<sycl::info::device::name>()
                      << std::endl;
    }
}

void list_default_device()
{
    sycl::queue queue(sycl::default_selector_v);
    std::cout << "Default device: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";
    return;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>>
syevd(std::vector<T> &a,
      const std::int64_t LDA,
      const std::int64_t N,
      const onemkl::job jobz = onemkl::job::V,
      const onemkl::uplo upper_lower = onemkl::uplo::U)
{
    sycl::queue queue(sycl::default_selector_v);

    T *a_ = sycl::malloc_device<T>(LDA * N, queue);
    T *w_ = sycl::malloc_device<T>(N, queue);

    queue.copy(a.data(), a_, LDA * N).wait();

    const std::int64_t lda = std::max<size_t>(1UL, LDA);
    const std::int64_t scratchpad_size =
        onemkl::lapack::syevd_scratchpad_size<T>(queue, jobz, upper_lower, N,
                                                 lda);
    T *scratchpad = nullptr;
    std::stringstream error_msg;
    std::int64_t info = 0;
    sycl::event syevd_event;

    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, queue);
        syevd_event = onemkl::lapack::syevd(
            queue,
            jobz, // 'jobz == job::vec' means eigenvalues and eigenvectors are
                  // computed.
            upper_lower, // 'upper_lower == job::upper' means the upper
                         // triangular part of A, or the lower triangular
                         // otherwise
            N,           // The order of the matrix A (0 <= n)
            a_,  // Pointer to A, size (lda, *), where the 2nd dimension,
                 // must be at least max(1, n) If 'jobz == job::vec', then
                 // on exit it will contain the eigenvectors of A
            lda, // The leading dimension of a, must be at least max(1, n)
            w_,  // Pointer to array of size at least n, it will contain
                 // the eigenvalues of A in ascending order
            scratchpad, // Pointer to scratchpad memory to be used by MKL
                        // routine for storing intermediate results
            scratchpad_size, {});
    } catch (onemkl::lapack::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during syevd() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
        info = e.info();
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during syevd() call:\n"
                  << e.what();
        info = -1;
    }

    if (info != 0) // an unexected error occurs
    {
        if (scratchpad != nullptr)
            sycl::free(scratchpad, queue);
        throw std::runtime_error(error_msg.str());
    }
    syevd_event.wait();

    T a_temp[LDA * N];
    T w_temp[N];

    queue.copy(a_, a_temp, LDA * N).wait();
    queue.copy(w_, w_temp, N).wait();

    sycl::free(a_, queue);
    sycl::free(w_, queue);

    std::vector<T> _a(a_temp, a_temp + (LDA * N));
    std::vector<T> w(w_temp, w_temp + N);

    return std::pair<std::vector<T>, std::vector<T>>(_a, w);
}

static void DPEX_LAPACK_eigh(arystruct_t *arystruct)
// static void DPEX_LAPACK_eigh()
{
    std::cout << "arystruct->nitems = " << (int)(arystruct->nitems)
              << std::endl;
    std::cout << "arystruct->itemsize = " << (int)(arystruct->itemsize)
              << std::endl;

    // double *data = (double *)(arystruct->data);
    // std::cout << "arystruct->data[0] = " << data[0] << std::endl;

    list_platforms();
    list_default_device();

    const std::int64_t LDA = 5;
    const std::int64_t N = LDA;

    std::vector<double> a = {6.39, 0.00,  0.00,  0.00,  0.00,  0.13,  8.37,
                             0.00, 0.00,  0.00,  -8.23, -4.46, -9.58, 0.00,
                             0.00, 5.71,  -6.10, -9.25, 3.72,  0.00,  -3.18,
                             7.21, -7.42, 8.54,  2.51};

    std::cout << "a:\n" << format(a, N, N, LDA) << std::endl << std::endl;

    std::pair<std::vector<double>, std::vector<double>> res =
        syevd<double>(a, LDA, N);
    std::vector<double> v = res.first;
    std::vector<double> w = res.second;

    std::cout << "v:\n" << format(v, N, N, LDA) << std::endl << std::endl;
    std::cout << "w:\n" << format(w, 1, N, 1) << std::endl;

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
