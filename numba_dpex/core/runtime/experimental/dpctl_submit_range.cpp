// SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "nrt_reserve_meminfo.h"

#include "_dbg_printer.h"
#include "syclinterface/dpctl_sycl_type_casters.hpp"
// #include "dpctl_error_handlers.h"
#include <CL/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace dpctl::syclinterface;

enum error_level : int
{
    none = 0,
    error = 1,
    warning = 2
};

void error_handler(const std::exception &e,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type = error_level::error);

void error_handler(const std::string &what,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type = error_level::warning);

int requested_verbosity_level(void)
{
    char *verbose = nullptr;

#ifdef _WIN32
    size_t len = 0;
    _dupenv_s(&verbose, &len, "DPCTL_VERBOSITY");
#else
    verbose = std::getenv("DPCTL_VERBOSITY");
#endif

    int requested_level = 0;

    if (verbose) {
        if (!std::strncmp(verbose, "none", 4))
            requested_level = error_level::none;
        else if (!std::strncmp(verbose, "error", 5))
            requested_level = error_level::error;
        else if (!std::strncmp(verbose, "warning", 7))
            requested_level = error_level::warning;
    }

#ifdef _WIN32
    if (verbose)
        free(verbose);
#endif

    return requested_level;
}

void output_message(std::string ss_str, error_level error_type)
{
#ifdef ENABLE_GLOG
    switch (error_type) {
    case error_level::error:
        LOG(ERROR) << "[ERR] " << ss_str;
        break;
    case error_level::warning:
        LOG(WARNING) << "[WARN] " << ss_str;
        break;
    default:
        LOG(FATAL) << "[FATAL] " << ss_str;
    }
#else
    switch (error_type) {
    case error_level::error:
        std::cerr << "[ERR] " << ss_str;
        break;
    case error_level::warning:
        std::cerr << "[WARN] " << ss_str;
        break;
    default:
        std::cerr << "[FATAL] " << ss_str;
    }
#endif
}

void error_handler(const std::exception &e,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type)
{
    int requested_level = requested_verbosity_level();
    int error_level = static_cast<int>(error_type);

    bool to_output = requested_level >= error_level;

    if (to_output) {
        std::stringstream ss;
        ss << e.what() << " in " << func_name << " at " << file_name << ":"
           << line_num << std::endl;

        output_message(ss.str(), error_type);
    }
}

void error_handler(const std::string &what,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type)
{
    int requested_level = requested_verbosity_level();
    int error_level = static_cast<int>(error_type);

    bool to_output = requested_level >= error_level;

    if (to_output) {
        std::stringstream ss;
        ss << what << " in " << func_name << " at " << file_name << ":"
           << line_num << std::endl;

        output_message(ss.str(), error_type);
    }
}

/*!
 * @brief Set the kernel arg object
 *
 * @param    cgh            My Param doc
 * @param    Arg            My Param doc
 */
bool set_kernel_arg(handler &cgh,
                    size_t idx,
                    __dpctl_keep void *Arg,
                    DPCTLKernelArgType ArgTy)
{
    bool arg_set = true;

    switch (ArgTy) {
    case DPCTL_CHAR:
        cgh.set_arg(idx, *(char *)Arg);
        break;
    case DPCTL_SIGNED_CHAR:
        cgh.set_arg(idx, *(signed char *)Arg);
        break;
    case DPCTL_UNSIGNED_CHAR:
        cgh.set_arg(idx, *(unsigned char *)Arg);
        break;
    case DPCTL_SHORT:
        cgh.set_arg(idx, *(short *)Arg);
        break;
    case DPCTL_INT:
        cgh.set_arg(idx, *(int *)Arg);
        break;
    case DPCTL_UNSIGNED_INT:
        cgh.set_arg(idx, *(unsigned int *)Arg);
        break;
    case DPCTL_UNSIGNED_INT8:
        cgh.set_arg(idx, *(uint8_t *)Arg);
        break;
    case DPCTL_LONG:
        cgh.set_arg(idx, *(long *)Arg);
        break;
    case DPCTL_UNSIGNED_LONG:
        cgh.set_arg(idx, *(unsigned long *)Arg);
        break;
    case DPCTL_LONG_LONG:
        // <---
        cgh.set_arg(idx, *(long long *)Arg);
        break;
    case DPCTL_UNSIGNED_LONG_LONG:
        cgh.set_arg(idx, *(unsigned long long *)Arg);
        break;
    case DPCTL_SIZE_T:
        cgh.set_arg(idx, *(size_t *)Arg);
        break;
    case DPCTL_FLOAT:
        cgh.set_arg(idx, *(float *)Arg);
        break;
    case DPCTL_DOUBLE:
        cgh.set_arg(idx, *(double *)Arg);
        break;
    case DPCTL_LONG_DOUBLE:
        cgh.set_arg(idx, *(long double *)Arg);
        break;
    case DPCTL_VOID_PTR:
        // <----
        cgh.set_arg(idx, Arg);
        break;
    default:
        arg_set = false;
        error_handler("Kernel argument could not be created.", __FILE__,
                      __func__, __LINE__);
        break;
    }
    return arg_set;
}

extern "C"
{

    DPCTLSyclEventRef
    DpexDPCTLQueue_SubmitRange(const DPCTLSyclKernelRef KRef,
                               const DPCTLSyclQueueRef QRef,
                               void **Args,
                               const DPCTLKernelArgType *ArgTypes,
                               size_t NArgs,
                               //    const size_t Range[3],
                               const size_t *Range,
                               size_t NDims,
                               const DPCTLSyclEventRef *DepEvents,
                               size_t NDepEvents)
    {
        auto Kernel = unwrap<kernel>(KRef);
        auto Queue = unwrap<queue>(QRef);
        event e;

        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-DEBUG: size of *void %d.\n",sizeof(void*)));
        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-DEBUG: size of size_t %d.\n",sizeof(size_t)));

        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-DEBUG: submit range %d (%d, %d, %d).\n",
                            NDims, Range[0], Range[1], Range[2]););

        std::cout << "debug from std: " << Range[0] << std::endl;

        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: arg size %d.\n", NArgs););

        // Args[0] = 9;

        try {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                for (auto i = 0ul; i < NDepEvents; ++i)
                    cgh.depends_on(*unwrap<event>(DepEvents[i]));

                for (auto i = 0ul; i < NArgs; ++i) {
                    // \todo add support for Sycl buffers
                    std::cout << "arg: " << i << " " << Args[i] << " "
                              << ArgTypes[i] << std::endl;
                    if (!set_kernel_arg(cgh, i, Args[i], ArgTypes[i]))
                        exit(1);
                }
                switch (NDims) {
                case 1:
                {
                    DPEXRT_DEBUG(
                        drt_debug_print("DPEXRT-DEBUG: submit range<1>.\n"));
                    auto rng = range<1>{Range[0]};
                    std::cout << "rng debug: " << rng[0] << " " << rng.size()
                              << std::endl;
                    cgh.parallel_for(rng, *Kernel);
                    DPEXRT_DEBUG(
                        drt_debug_print("DPEXRT-DEBUG: sent range<1>.\n"));
                    break;
                }
                case 2:
                    DPEXRT_DEBUG(
                        drt_debug_print("DPEXRT-DEBUG: submit range<2>.\n"));
                    cgh.parallel_for(range<2>{Range[0], Range[1]}, *Kernel);
                    break;
                case 3:
                    DPEXRT_DEBUG(
                        drt_debug_print("DPEXRT-DEBUG: submit range<3>.\n"));
                    cgh.parallel_for(range<3>{Range[0], Range[1], Range[2]},
                                     *Kernel);
                    break;
                default:
                    throw std::runtime_error(
                        "Range cannot be greater than three "
                        "dimensions.");
                }
            });
        } catch (std::exception const &e) {
            DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: faced exception %s.\n",
                                         e.what()));
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }

        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: force wait"));
        e.wait();

        return wrap<event>(new event(std::move(e)));
    }

    DPCTLSyclEventRef DpexDPCTLQueue_SubmitNDRange(
        const DPCTLSyclKernelRef KRef,
        const DPCTLSyclQueueRef QRef,
        void **Args,
        const DPCTLKernelArgType *ArgTypes,
        size_t NArgs,
        const size_t gRange[3],
        const size_t lRange[3],
        size_t NDims,
        const DPCTLSyclEventRef *DepEvents,
        size_t NDepEvents)
    {
        auto Kernel = unwrap<kernel>(KRef);
        auto Queue = unwrap<queue>(QRef);
        event e;

        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: submit nd range %d (%d, "
                                     "%d, %d), (%d, %d, %d).\n",
                                     NDims, gRange[0], gRange[1], gRange[2],
                                     lRange[0], lRange[1], lRange[2]););

        try {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                if (DepEvents)
                    for (auto i = 0ul; i < NDepEvents; ++i) {
                        auto ei = unwrap<event>(DepEvents[i]);
                        if (ei)
                            cgh.depends_on(*ei);
                    }

                for (auto i = 0ul; i < NArgs; ++i) {
                    // \todo add support for Sycl buffers
                    if (!set_kernel_arg(cgh, i, Args[i], ArgTypes[i]))
                        exit(1);
                }
                switch (NDims) {
                case 1:
                    cgh.parallel_for(nd_range<1>{{gRange[0]}, {lRange[0]}},
                                     *Kernel);
                    break;
                case 2:
                    cgh.parallel_for(nd_range<2>{{gRange[0], gRange[1]},
                                                 {lRange[0], lRange[1]}},
                                     *Kernel);
                    break;
                case 3:
                    cgh.parallel_for(
                        nd_range<3>{{gRange[0], gRange[1], gRange[2]},
                                    {lRange[0], lRange[1], lRange[2]}},
                        *Kernel);
                    break;
                default:
                    throw std::runtime_error(
                        "Range cannot be greater than three "
                        "dimensions.");
                }
            });
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }

        return wrap<event>(new event(std::move(e)));
    }
}
