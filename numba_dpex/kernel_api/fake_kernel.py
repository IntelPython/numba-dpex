# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from collections import namedtuple
from collections.abc import Iterable
from functools import reduce
from itertools import product

import numpy as np

try:
    from greenlet import greenlet

    _greenlet_found = True
except ImportError:
    _greenlet_found = False

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


class fake_atomic:
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


class fake_local:
    @staticmethod
    def array(shape, dtype):
        return simulator_impl.local_array(shape, dtype)


def fake_mem_fence(flags):
    pass  # Nothing


class fake_private:
    @staticmethod
    def array(shape, dtype):
        return simulator_impl.private_array(shape, dtype)


class fake_group:
    @staticmethod
    def reduce_add(value):
        return simulator_impl.group_reduce(value, lambda a, b: a + b)


class fake_numba_dpex:
    @staticmethod
    def get_global_id(id):
        return simulator_impl.get_global_id(id)

    def get_local_id(id):
        return simulator_impl.get_local_id(id)


def fake_barrier(flags):
    simulator_impl.barrier()


class ExecutionState:
    def __init__(self, global_size, local_size):
        self._global_size = global_size
        self._local_size = local_size
        self._indices = [0] * len(self._global_size)
        self._wg_size = [None]
        self._tasks = []
        self._current_task = [None]
        self._local_arrays = []
        self._current_local_array = [0]
        self._reduce_val = [None]

    # def get(self):
    #     return self._state        # noqa: E800

    # def _save(self):
    #     indices = copy.deepcopy(self._state.indices)              # noqa: E800
    #     current_local_array = self._state.current_local_array[0]  # noqa: E800
    #     self._state.current_local_array[0] = 0                    # noqa: E800
    #     return (indices, current_local_array)                     # noqa: E800

    # def _restore(self, state):
    #     self._state.indices[:] = state[0]                 # noqa: E800
    #     self._state.current_local_array[0] = state[1]     # noqa: E800

    def _reset(self, wg_size):
        self._wg_size[0] = wg_size
        self._current_task[0] = 0
        self._local_arrays.clear()
        self._current_local_array[0] = 0

    def _barrier(self):
        # wg_size = state.wg_size[0]            # noqa: E800
        assert self._wg_size > 0
        if self._wg_size > 1:
            assert len(self._tasks) > 0
            # saved_state = self._save(state)   # noqa: E800
            next_task = self._current_task[0] + 1
            if next_task >= self._wg_size:
                next_task = 0
            self._current_task[0] = next_task
            self._tasks[next_task].switch()
            # self._restore(state, saved_state) # noqa: E800

    def _reduce(self, value, op):
        if self._current_task[0] == 0:
            self._reduce_val[0] = value
        else:
            self._reduce_val[0] = op(self._reduce_val[0], value)
        # _barrier_impl(state)      # noqa: E800
        self._barrier()
        return self._reduce_val[0]

    def _destroy(self):
        self._indices = [0] * len(self._global_size)
        self._wg_size = [None]
        self._tasks = []
        self._current_task = [None]
        self._local_arrays = []
        self._current_local_array = [0]
        self._reduce_val = [None]

    def _capture_func(self, func, indices, args):
        def wrapper():
            self._indices[:] = indices
            func(*args)

        return wrapper


# TODO: Share code with dispatcher
class FakeKernel:
    def __init__(self, func):
        self._func = func
        self._globals_to_replace = [
            (numba_dpex, fake_numba_dpex),
            (get_global_id, self._get_global_id),
            (get_local_id, self._get_local_id),
            (get_group_id, self._get_group_id),
            (get_global_size, self._get_global_size),
            (get_local_size, self._get_local_size),
            (atomic, fake_atomic),
            (barrier, fake_barrier),
            (mem_fence, fake_mem_fence),
            (local, fake_local),
            (local.array, fake_local.array),
            (private, fake_private),
            (private.array, fake_private.array),
            # (group, group_proxy), # noqa: E800
            # (group.reduce_add, group_proxy.reduce_add),   # noqa: E800
        ]
        self._need_barrier = self._has_barrier_ops()
        self._execution_state = None
        self._saved_globals = self._replace_globals(self._func.__globals__)
        self._saved_closure = self._replace_closure(self._func.__closure__)

    def _has_barrier_ops(self):
        for v in self._func.__globals__.values():
            if v is barrier:
                return True
        return False

    def _replace_global_func(self, global_obj):
        for old_val, new_val in self._globals_to_replace:
            if global_obj is old_val:
                return new_val
        return global_obj

    def _replace_globals(self, src):
        old_globals = list(src.items())
        for name, val in src.items():
            src[name] = self._replace_global_func(val)
        return old_globals

    def _restore_globals(self, src):
        src.update(self._saved_globals)

    def _replace_closure(self, src):
        if src is None:
            return None

        old_vals = [e.cell_contents for e in src]
        for e in src:
            old_val = e.cell_contents
            e.cell_contents = self._replace_global_func(old_val)
        return old_vals

    def _restore_closure(self, src):
        if self._saved_closure is None:
            return

        for i in range(len(src)):
            src[i].cell_contents = self._saved_closure[i]

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

    def _execute(self, global_size, local_size, args):
        if local_size is None or len(local_size) == 0:
            local_size = (1,) * len(global_size)

        # saved_globals = _replace_globals(func.__globals__, replace_global_func)   # noqa: E800
        # saved_closure = _replace_closure(func.__closure__, replace_global_func)   # noqa: E800
        # state = _setup_execution_state(global_size, local_size)   # noqa: E800
        assert self._execution_state is None
        self._execution_state = ExecutionState(global_size, local_size)
        try:
            groups = tuple(
                (g + l - 1) // l for g, l in zip(global_size, local_size)
            )
            for gid in product(*(range(g) for g in groups)):
                offset = tuple(g * l for g, l in zip(gid, local_size))
                size = tuple(
                    min(g - o, l)
                    for o, g, l in zip(offset, global_size, local_size)
                )
                count = reduce(lambda a, b: a * b, size)
                self._execution_state._reset(count)

                indices_range = (range(o, o + s) for o, s in zip(offset, size))

                if self._need_barrier:
                    global _greenlet_found
                    assert _greenlet_found, "greenlet package not installed"
                    # tasks = self._execution_state.tasks   # noqa: E800
                    assert len(self._execution_state._tasks) == 0
                    for indices in product(*indices_range):
                        self._execution_state._tasks.append(
                            greenlet(
                                self._execution_state._capture_func(
                                    self._func, indices, args
                                )
                            )
                        )

                    for t in self._execution_state._tasks:
                        t.switch()

                    self._execution_state._tasks.clear()
                else:
                    for indices in product(*indices_range):
                        self._execution_state._indices[:] = indices
                        self._func(*args)

        finally:
            self._restore_closure(self._func.__closure__)
            self._restore_globals(self._func.__globals__)

    def _get_global_id(self, index):
        return self._execution_state._indices[index]

    def _get_local_id(self, index):
        return (
            self._execution_state._indices[index]
            % self._execution_state._local_size[index]
        )

    def _get_group_id(self, index):
        return (
            self._execution_state._indices[index]
            // self._execution_state._local_size[index]
        )

    def _get_global_size(self, index):
        return self._execution_state._global_size[index]

    def _get_local_size(self, index):
        return self._execution_state._local_size[index]

    # def __call__(self, *args, **kwargs):
    #     # need_barrier = self._have_barrier_ops()       # noqa: E800
    #     self._execute(
    #         self._global_range[::-1],                     # noqa: E800
    #         None if self._local_range is None else self._local_range[::-1],   # noqa: E800
    #         self._func,
    #         args,
    #         # need_barrier=self._need_barrier,    # noqa: E800
    #         # replace_global_func=self._replace_global_func,  # noqa: E800
    #     ) # noqa: E800


# def kernel(func):
#     return FakeKernel(func)   # noqa: E800
