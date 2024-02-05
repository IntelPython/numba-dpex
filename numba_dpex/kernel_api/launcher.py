# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from collections import namedtuple
from collections.abc import Iterable
from functools import reduce
from itertools import product

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


_ExecutionState = namedtuple(
    "_ExecutionState",
    [
        "global_size",
        "local_size",
        "indices",
        "wg_size",
        "tasks",
        "current_task",
        "local_arrays",
        "current_local_array",
        "reduce_val",
    ],
)


class ExecutionState:
    def __init__(self, global_size, local_size):
        self._state = _ExecutionState(
            global_size=global_size,
            local_size=local_size,
            indices=[0] * len(global_size),
            wg_size=[None],
            tasks=[],
            current_task=[None],
            local_arrays=[],
            current_local_array=[0],
            reduce_val=[None],
        )

    def get(self):
        return self._state

    # def _save(self):
    #     indices = copy.deepcopy(self._state.indices)              # noqa: E800
    #     current_local_array = self._state.current_local_array[0]  # noqa: E800
    #     self._state.current_local_array[0] = 0                    # noqa: E800
    #     return (indices, current_local_array)                     # noqa: E800

    # def _restore(self, state):
    #     self._state.indices[:] = state[0]                 # noqa: E800
    #     self._state.current_local_array[0] = state[1]     # noqa: E800

    def _reset(self, wg_size):
        self._state.wg_size[0] = wg_size
        self._state.current_task[0] = 0
        self._state.local_arrays.clear()
        self._state.current_local_array[0] = 0

    def _barrier(self):
        # wg_size = state.wg_size[0]            # noqa: E800
        assert self._state.wg_size > 0
        if self._state.wg_size > 1:
            assert len(self._state.tasks) > 0
            # saved_state = self._save(state)   # noqa: E800
            next_task = self._state.current_task[0] + 1
            if next_task >= self._state.wg_size:
                next_task = 0
            self._state.current_task[0] = next_task
            self._state.tasks[next_task].switch()
            # self._restore(state, saved_state) # noqa: E800

    def _reduce(self, value, op):
        if self._state.current_task[0] == 0:
            self._state.reduce_val[0] = value
        else:
            self._state.reduce_val[0] = op(self._state.reduce_val[0], value)
        # _barrier_impl(state)      # noqa: E800
        self._barrier()
        return self._state.reduce_val[0]

    def _destroy(self):
        self._state = None

    def _replace_globals(self, src, replace_global_func):
        old_globals = list(src.items())
        for name, val in src.items():
            src[name] = replace_global_func(val)
        return old_globals


# TODO: Share code with dispatcher
class FakeKernel:
    def __init__(self, func):
        self._func = func
        self._barrier_ops = [barrier]
        self._globals_to_replace = [
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
        self._need_barrier = self._has_barrier_ops()

    def _has_barrier_ops(self):
        for v in self._func.__globals__.values():
            for b in self._barrier_ops:
                if v is b:
                    return True
        return False

    def _replace_global_func(self, global_obj):
        for old_val, new_val in self._globals_to_replace:
            if global_obj is old_val:
                return new_val
        return global_obj

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
        # need_barrier = self._have_barrier_ops()       # noqa: E800
        simulator_impl.execute_kernel(
            self._global_range[::-1],
            None if self._local_range is None else self._local_range[::-1],
            self._func,
            args,
            need_barrier=self._need_barrier,
            replace_global_func=self._replace_global_func,
        )


def kernel(func):
    return FakeKernel(func)
