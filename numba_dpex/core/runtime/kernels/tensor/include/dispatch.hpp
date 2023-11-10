// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file dispatch.hpp
/// \brief Implements the "function dispatch vector".
///
/// Implements a so called "function dispatch vector". A function dispatch
/// vector is an array of (templated) function pointers that are specialized
/// to a given data type. Since the data type of the tensor is not known
/// beforehand, we keep an array of functions specialized to the 14 types in
/// ``dpex::rt::kernel::tensor::typeutils::typenum_t``.
///
//===----------------------------------------------------------------------===//

#ifndef __DISPATCH_HPP__
#define __DISPATCH_HPP__

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

namespace dpex
{
namespace rt
{
namespace kernel
{
namespace tensor
{
namespace dispatch
{

/**
 * \brief The class to implement a "dispatch vector"
 *
 * Implements a so called "function dispatch vector". A function dispatch
 * vector is an array of (templated) function pointers that are specialized
 * to a given data type. Since the data type of the tensor is not known
 * beforehand, we keep an array of functions specialized to the 14 types in
 * ``dpex::rt::kernel::tensor::typeutils::typenum_t``. Each function instance
 * is accessed using a predefined indices that point to the specific data type
 * a function was already specialized to.
 *
 * \tparam funcPtrT     The pointer to the function.
 * \tparam factory      A factory function that returns a specialized function
 *                      pointer.
 * \tparam _num_types   A constant type to indicate the total number of data
 *                      types in ``typenum_t``.
 */
template <typename funcPtrT,
          template <typename fnT, typename T>
          typename factory,
          int _num_types>
class DispatchVectorBuilder
{
private:
    /**
     * \brief   Use the 'factory' to return a function specialized to type `Ty`
     *
     * The 'factories' are defined in
     *
     * \tparam Ty               The type the function would be specialized.
     * \return const funcPtrT   The pointer to the function from the factory.
     */
    template <typename Ty> const funcPtrT func_per_type() const
    {
        funcPtrT f = factory<funcPtrT, Ty>{}.get();
        return f;
    }

public:
    DispatchVectorBuilder() = default;
    ~DispatchVectorBuilder() = default;

    /**
     * \brief
     *
     * \param vector
     */
    void populate_dispatch_vector(funcPtrT vector[]) const
    {
        const auto fn_map_by_type = {
            func_per_type<bool>(), // 0
            func_per_type<int8_t>(),
            func_per_type<uint8_t>(),
            func_per_type<int16_t>(),
            func_per_type<uint16_t>(),
            func_per_type<int32_t>(), // 5
            func_per_type<uint32_t>(),
            func_per_type<int64_t>(),
            func_per_type<uint64_t>(),
            func_per_type<sycl::half>(),
            func_per_type<float>(), // 10
            func_per_type<double>(),
            func_per_type<std::complex<float>>(),
            func_per_type<std::complex<double>>() // 13
        };
        assert(fn_map_by_type.size() == _num_types);
        int ty_id = 0;
        for (auto &fn : fn_map_by_type) {
            vector[ty_id] = fn;
            ++ty_id;
        }
    }
};

} // namespace dispatch
} // namespace tensor
} // namespace kernel
} // namespace rt
} // namespace dpex

#endif
