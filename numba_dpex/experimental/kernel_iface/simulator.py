# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Iterable

import numba_dpex
from numba_dpex import (
    NdRange,
    Range,
    atomic,
    barrier,
    get_global_id,
    get_global_size,
    get_group_id,
    get_local_id,
    get_local_size,
    local,
    mem_fence,
    private,
)
from numba_dpex.core.exceptions import (
    IllegalRangeValueError,
    InvalidKernelLaunchArgsError,
)

from . import simulator_impl


class atomic_proxy:
    @staticmethod
    def add(arr, ind, val):
        new_val = arr[ind] + val
        arr[ind] = new_val
        return new_val

    @staticmethod
    def sub(arr, ind, val):
        new_val = arr[ind] - val
        arr[ind] = new_val
        return new_val


def mem_fence_proxy(flags):
    pass  # Nothing


class local_proxy:
    @staticmethod
    def array(shape, dtype):
        return simulator_impl.local_array(shape, dtype)


class private_proxy:
    @staticmethod
    def array(shape, dtype):
        return simulator_impl.private_array(shape, dtype)


class group_proxy:
    @staticmethod
    def reduce_add(value):
        return simulator_impl.group_reduce(value, lambda a, b: a + b)


class dpex_proxy:
    @staticmethod
    def get_global_id(id):
        return simulator_impl.get_global_id(id)

    def get_local_id(id):
        return simulator_impl.get_local_id(id)


def barrier_proxy(flags):
    simulator_impl.barrier()


_globals_to_replace = [
    (numba_dpex, dpex_proxy),
    (get_global_id, simulator_impl.get_global_id),
    (get_local_id, simulator_impl.get_local_id),
    (get_group_id, simulator_impl.get_group_id),
    (get_global_size, simulator_impl.get_global_size),
    (get_local_size, simulator_impl.get_local_size),
    (atomic, atomic_proxy),
    (barrier, barrier_proxy),
    (mem_fence, mem_fence_proxy),
    (local, local_proxy),
    (local.array, local_proxy.array),
    (private, private_proxy),
    (private.array, private_proxy.array),
    # (group, group_proxy), # noqa: E800
    # (group.reduce_add, group_proxy.reduce_add),   # noqa: E800
]

_barrier_ops = [barrier]


def _have_barrier_ops(func):
    for v in func.__globals__.values():
        for b in _barrier_ops:
            if v is b:
                return True
    return False


def _replace_global_func(global_obj):
    for old_val, new_val in _globals_to_replace:
        if global_obj is old_val:
            return new_val

    return global_obj


# TODO: Share code with dispatcher
class Kernel:
    def __init__(self, func):
        self._func = func

    def __getitem__(self, args):
        if isinstance(args, Range):
            # we need inversions, see github issue #889
            self._global_range = list(args)[::-1]
            self._local_range = None
        elif isinstance(args, NdRange):
            # we need inversions, see github issue #889
            self._global_range = list(args.global_range)[::-1]
            self._local_range = list(args.local_range)[::-1]
        else:
            if (
                isinstance(args, tuple)
                and len(args) == 2
                and isinstance(args[0], int)
                and isinstance(args[1], int)
            ):
                self._global_range = [args[0]]
                self._local_range = [args[1]]
                return self

            args = [args] if not isinstance(args, Iterable) else args
            nargs = len(args)

            # Check if the kernel enquing arguments are sane
            if nargs < 1 or nargs > 2:
                raise InvalidKernelLaunchArgsError(kernel_name=self.kernel_name)

            g_range = (
                [args[0]] if not isinstance(args[0], Iterable) else args[0]
            )
            # If the optional local size argument is provided
            l_range = None
            if nargs == 2:
                if args[1] != []:
                    l_range = (
                        [args[1]]
                        if not isinstance(args[1], Iterable)
                        else args[1]
                    )

            if len(g_range) < 1:
                raise IllegalRangeValueError(kernel_name=self.kernel_name)

            self._global_range = list(g_range)[::-1]
            self._local_range = list(l_range)[::-1] if l_range else None

        return self

    def __call__(self, *args, **kwargs):
        need_barrier = _have_barrier_ops(self._func)
        simulator_impl.execute_kernel(
            self._global_range[::-1],
            None if self._local_range is None else self._local_range[::-1],
            self._func,
            args,
            need_barrier=need_barrier,
            replace_global_func=_replace_global_func,
        )


def kernel(func):
    return Kernel(func)
