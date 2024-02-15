# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation of mock kernel launcher functions
"""

from inspect import signature
from itertools import product

from .index_space_ids import Group, Item, NdItem
from .ranges import NdRange, Range


def _range_kernel_launcher(kernel_fn, index_range, *kernel_args):
    """Executes a function that mocks a range kernel.

    Converts the range into a set of index tuple that represent an element in
    the iteration domain over which the kernel will be executed. Then the
    kernel is called sequentially over that set of indices after each index
    value is used to construct an Item object.

    Args:
        kernel_fn : A callable function object
        index_range (numba_dpex.Range): An instance of a Range object

    Raises:
        ValueError: If the number of passed in kernel arguments is not the
        number of function parameters subtracted by one. The first kernel
        argument is expected to be an Item object.
    """

    range_sets = [range(ir) for ir in index_range]
    index_tuples = list(product(*range_sets))

    for idx in index_tuples:
        it = Item(extent=index_range, index=idx)

        if len(signature(kernel_fn).parameters) - len(kernel_args) != 1:
            raise ValueError(
                "Required number of kernel function arguments do not "
                "match provided number of kernel args"
            )

        kernel_fn(it, *kernel_args)


def _ndrange_kernel_launcher(kernel_fn, index_range, *kernel_args):
    """Executes a function that mocks a range kernel.

    Args:
        kernel_fn : A callable function object
        index_range (numba_dpex.NdRange): An instance of a NdRange object

    Raises:
        ValueError: If the number of passed in kernel arguments is not the
        number of function parameters subtracted by one. The first kernel
        argument is expected to be an Item object.
    """
    group_range = tuple(
        gr // lr
        for gr, lr in zip(index_range.global_range, index_range.local_range)
    )
    local_range_sets = [range(ir) for ir in index_range.local_range]
    group_range_sets = [range(gr) for gr in group_range]
    local_index_tuples = list(product(*local_range_sets))
    group_index_tuples = list(product(*group_range_sets))

    # Loop over the groups (parallel loop)
    for gidx in group_index_tuples:
        # loop over work items in the group (parallel loop)
        for lidx in local_index_tuples:
            global_id = []
            # to calculate global indices
            for dim, gidx_val in enumerate(gidx):
                global_id.append(
                    gidx_val * index_range.local_range[dim] + lidx[dim]
                )
            # Every NdItem has its own global Item, local Item and Group
            nditem = NdItem(
                global_item=Item(
                    extent=index_range.global_range, index=global_id
                ),
                local_item=Item(extent=index_range.local_range, index=lidx),
                group=Group(
                    index_range.global_range,
                    index_range.local_range,
                    group_range,
                    gidx,
                ),
            )

            if len(signature(kernel_fn).parameters) - len(kernel_args) != 1:
                raise ValueError(
                    "Required number of kernel function arguments do not "
                    "match provided number of kernel args"
                )

            kernel_fn(nditem, *kernel_args)


def call_kernel(kernel_fn, index_range, *kernel_args):
    """Mocks the launching of a kernel function over either a Range or NdRange.

    Args:
        kernel_fn : A callable function object
        index_range (numba_dpex.Range): An instance of a Range object

    Raises:
        ValueError: If the first positional argument is not callable
        ValueError: If the second positional argument is not a Range or an
        Ndrange object
    """
    if not callable(kernel_fn):
        raise ValueError(
            "Expected the first positional argument to be a function object"
        )
    if isinstance(index_range, Range):
        _range_kernel_launcher(kernel_fn, index_range, *kernel_args)
    elif isinstance(index_range, NdRange):
        _ndrange_kernel_launcher(kernel_fn, index_range, *kernel_args)
    else:
        raise ValueError(
            "Expected second positional argument to be Range or NdRange object"
        )
