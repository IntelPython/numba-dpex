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
    static void DPEX_ONEMKL_LAPACK_syevd(arystruct_t *as_a,
                                         arystruct_t *as_v,
                                         arystruct_t *as_w,
                                         std::int64_t lda,
                                         std::int64_t n,
                                         std::int64_t uplo);
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
std::string fmtcgvec(std::vector<T> &a, int m, int n, int lda)
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

template <typename T> std::string fmtcgarr(T *a, int m, int n, int lda)
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
void _syevd(sycl::queue queue,
            T *a,
            T *w,
            const std::int64_t LDA,
            const std::int64_t N,
            const onemkl::uplo upper_lower = onemkl::uplo::U)
{
    const onemkl::job jobz = onemkl::job::V;
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
            a,   // Pointer to A, size (lda, *), where the 2nd dimension,
                 // must be at least max(1, n) If 'jobz == job::vec', then
                 // on exit it will contain the eigenvectors of A
            lda, // The leading dimension of a, must be at least max(1, n)
            w,   // Pointer to array of size at least n, it will contain
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
    queue.wait();
}

static void DPEX_ONEMKL_LAPACK_syevd(arystruct_t *as_a,
                                     arystruct_t *as_v,
                                     arystruct_t *as_w,
                                     std::int64_t lda,
                                     std::int64_t n,
                                     std::int64_t uplo)
{
    list_platforms();
    list_default_device();

    sycl::queue queue(sycl::default_selector_v);

    const std::int64_t LDA = lda;
    const std::int64_t N = n;
    const onemkl::uplo upper_lower =
        (uplo == 1) ? onemkl::uplo::U : onemkl::uplo::L;

    double *a_ = (double *)(as_a->data);
    double *w_ = (double *)(as_w->data);
    double *v_ = (double *)(as_v->data);

    // double *a_ = (double *)sycl::malloc_device(LDA * N, queue);
    // double *w_ = (double *)sycl::malloc_device(N, queue);
    // double *v_ = (double*)sycl::malloc_device(LDA * N, queue);

    // queue.memcpy(a_, (double *)(as_a->data), sizeof(double) * LDA *
    // N).wait(); queue.memcpy(w_, (double *)(as_w->data), sizeof(double) *
    // N).wait(); queue.memcpy(v_, (double*)(as_v->data), sizeof(double) * LDA *
    // N).wait();

    _syevd<double>(queue, a_, w_, LDA, N, upper_lower);

    queue.copy(a_, v_, LDA * N).wait();
    // queue.memcpy((double *)(as_v->data), a_, sizeof(double) * LDA *
    // N).wait(); queue.memcpy((double *)(as_w->data), w_, sizeof(double) *
    // N).wait();

    // sycl::free(a_, queue);
    // sycl::free(w_, queue);
    // sycl::free(v_, queue);

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

    _declpointer("DPEX_ONEMKL_LAPACK_syevd",
                 (void *)(&DPEX_ONEMKL_LAPACK_syevd));

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

    PyModule_AddObject(m, "DPEX_ONEMKL_LAPACK_syevd",
                       PyLong_FromVoidPtr((void *)(&DPEX_ONEMKL_LAPACK_syevd)));
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
