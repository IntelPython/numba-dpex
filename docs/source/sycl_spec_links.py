# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Links to the SYCL 2020 specification that are used in docstring.

The module provides a dictionary in the format needed by the sphinx.ext.extlinks
extension.
"""

sycl_ext_links = {
    "sycl_item": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class%s",
        None,
    ),
    "sycl_group": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#group-class%s",
        None,
    ),
    "sycl_nditem": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:nditem.class%s",
        None,
    ),
    "sycl_ndrange": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsubsec:nd-range-class%s",
        None,
    ),
    "sycl_range": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class%s",
        None,
    ),
    "sycl_atomic_ref": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:atomic-references%s",
        None,
    ),
    "sycl_local_accessor": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:accessor.local%s",
        None,
    ),
    "sycl_private_memory": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_parallel_for_hierarchical_invoke%s",
        None,
    ),
    "sycl_memory_scope": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:memory-scope%s",
        None,
    ),
    "sycl_memory_order": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:memory-ordering%s",
        None,
    ),
    "sycl_addr_space": (
        "https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_address_space_classes%s",
        None,
    ),
}
