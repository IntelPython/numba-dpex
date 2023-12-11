// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Based on Peter Dimov's proposal
//  http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2005/n1756.pdf
//  issue 6.18.
//
//  This also contains public domain code from MurmurHash. From the
//  MurmurHash header:

// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// 2023 Intel Corporation
// Copied hash_combine and hash_combine_impl from boost
// (https://www.boost.org/doc/libs/1_76_0/boost/container_hash/hash.hpp) and
// changed hash_combine to use std::hash<T> instead of boost::hash<T>.

#include <functional>

namespace boost
{
namespace hash_detail
{
template <typename SizeT>
inline void hash_combine_impl(SizeT &seed, SizeT value)
{
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace hash_detail

template <class T> inline void hash_combine(std::size_t &seed, T const &v)
{
    std::hash<T> hasher;
    return boost::hash_detail::hash_combine_impl(seed, hasher(v));
}
} // namespace boost
