# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


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
from numba_dpex.core.exceptions import UnsupportedKernelArgumentError


class ExecutionState:
    def __init__(self, global_size, local_size):
        print("ExecutionState.__init__()")
        self._global_size = global_size
        self._local_size = local_size
        self._indices = [0] * len(self._global_size)
        self._wg_size = [None]
        self._tasks = []
        self._current_task = [None]
        self._local_arrays = []
        self._current_local_array = [0]
        self._reduce_val = [None]

    def _reset(self, wg_size):
        print("ExecutionState._reset()")
        self._wg_size[0] = wg_size
        self._current_task[0] = 0
        self._local_arrays.clear()
        self._current_local_array[0] = 0

    def _barrier(self):
        print("ExecutionState._barrier()")
        assert self._wg_size > 0
        if self._wg_size > 1:
            assert len(self._tasks) > 0
            next_task = self._current_task[0] + 1
            if next_task >= self._wg_size:
                next_task = 0
            self._current_task[0] = next_task
            self._tasks[next_task].switch()

    def _reduce(self, value, op):
        print("ExecutionState._reduce()")
        if self._current_task[0] == 0:
            self._reduce_val[0] = value
        else:
            self._reduce_val[0] = op(self._reduce_val[0], value)
        self._barrier()
        return self._reduce_val[0]

    def _destroy(self):
        print("ExecutionState._destroy()")
        self._indices = [0] * len(self._global_size)
        self._wg_size = [None]
        self._tasks = []
        self._current_task = [None]
        self._local_arrays = []
        self._current_local_array = [0]
        self._reduce_val = [None]

    def _capture_func(self, func, indices, args):
        print("ExecutionState._capture_func()")

        def wrapper():
            self._indices[:] = indices
            func(*args)

        return wrapper


# TODO: Share code with dispatcher
class FakeKernel:
    _execution_state = None

    def __init__(
        self,
        func,
        index_space,
        *kernel_args,
    ):
        print("FakeKernel.__init__()")
        self._func = func
        self._global_range, self._local_range = self._get_ranges(index_space)
        self._kernel_args = kernel_args

        self._globals_to_replace = [
            (numba_dpex, FakeKernel),
            (get_global_id, FakeKernel.get_global_id),
            (get_local_id, FakeKernel.get_local_id),
            (get_group_id, FakeKernel.get_group_id),
            (get_global_size, FakeKernel.get_global_size),
            (get_local_size, FakeKernel.get_local_size),
            (atomic, FakeKernel.atomic),
            (barrier, FakeKernel.barrier),
            (mem_fence, FakeKernel.mem_fence),
            (local, FakeKernel.local),
            (local.array, FakeKernel.local.array),
            (private, FakeKernel.private),
            (private.array, FakeKernel.private.array),
            # (group, group_proxy), # noqa: E800
            # (group.reduce_add, group_proxy.reduce_add),   # noqa: E800
        ]

        self._need_barrier = self._has_barrier_ops()
        FakeKernel._execution_state = ExecutionState(
            self._global_range, self._local_range
        )

        self._saved_globals = self._replace_globals(self._func.__globals__)
        self._saved_closure = self._replace_closure(self._func.__closure__)

    def _get_ranges(self, index_space):
        print("FakeKernel._get_ranges()")
        if isinstance(index_space, Range):
            _global_range = list(index_space)
            _local_range = None
        elif isinstance(index_space, NdRange):
            _global_range = list(index_space.global_range)
            _local_range = list(index_space.local_range)
        else:
            raise UnsupportedKernelArgumentError(
                type=type(index_space),
                value=index_space,
                kernel_name=self._func.__name__,
            )
        if _local_range is None or len(_local_range) == 0:
            _local_range = (1,) * len(_global_range)
        return _global_range, _local_range

    def _has_barrier_ops(self):
        print("FakeKernel._has_barrier_ops()")
        for v in self._func.__globals__.values():
            if v is barrier:
                return True
        return False

    def _replace_global_func(self, global_obj):
        print("FakeKernel._replace_global_func()")
        for old_val, new_val in self._globals_to_replace:
            if global_obj is old_val:
                return new_val
        return global_obj

    def _replace_globals(self, src):
        print("FakeKernel._replace_globals()")
        old_globals = list(src.items())
        for name, val in src.items():
            src[name] = self._replace_global_func(val)
        return old_globals

    def _restore_globals(self, src):
        print("FakeKernel._restore_globals()")
        if self._saved_globals is None:
            return
        src.update(self._saved_globals)

    def _replace_closure(self, src):
        print("FakeKernel._replace_closure()")
        if src is None:
            return None

        old_vals = [e.cell_contents for e in src]
        for e in src:
            old_val = e.cell_contents
            e.cell_contents = self._replace_global_func(old_val)
        return old_vals

    def _restore_closure(self, src):
        print("FakeKernel._restore_closure()")
        if self._saved_closure is None:
            return

        for i in range(len(src)):
            src[i].cell_contents = self._saved_closure[i]

    def execute(self):
        print("FakeKernel.execute()")
        assert FakeKernel._execution_state
        try:
            groups = tuple(
                (g + l - 1) // l
                for g, l in zip(self._global_range, self._local_range)
            )
            for gid in product(*(range(g) for g in groups)):
                offset = tuple(g * l for g, l in zip(gid, self._local_range))
                size = tuple(
                    min(g - o, l)
                    for o, g, l in zip(
                        offset, self._global_range, self._local_range
                    )
                )
                count = reduce(lambda a, b: a * b, size)
                FakeKernel._execution_state._reset(count)

                indices_range = (range(o, o + s) for o, s in zip(offset, size))

                if self._need_barrier:
                    global _greenlet_found
                    assert _greenlet_found, "greenlet package not installed"
                    # tasks = self._execution_state.tasks   # noqa: E800
                    assert len(FakeKernel._execution_state._tasks) == 0
                    for indices in product(*indices_range):
                        FakeKernel._execution_state._tasks.append(
                            greenlet(
                                FakeKernel._execution_state._capture_func(
                                    self._func, indices, self._kernel_args
                                )
                            )
                        )

                    for t in FakeKernel._execution_state._tasks:
                        t.switch()

                    FakeKernel._execution_state._tasks.clear()
                else:
                    for indices in product(*indices_range):
                        FakeKernel._execution_state._indices[:] = indices
                        self._func(*self._kernel_args)

        finally:
            self._restore_closure(self._func.__closure__)
            self._restore_globals(self._func.__globals__)

    @staticmethod
    def get_global_id(index):
        print("FakeKernel.get_global_id()")
        return FakeKernel._execution_state._indices[index]

    @staticmethod
    def get_local_id(index):
        print("FakeKernel.get_local_id()")
        return (
            FakeKernel._execution_state._indices[index]
            % FakeKernel._execution_state._local_size[index]
        )

    @staticmethod
    def get_group_id(index):
        print("FakeKernel.get_group_id()")
        return (
            FakeKernel._execution_state._indices[index]
            // FakeKernel._execution_state._local_size[index]
        )

    @staticmethod
    def get_global_size(index):
        print("FakeKernel.get_global_size()")
        return FakeKernel._execution_state._global_size[index]

    @staticmethod
    def get_local_size(index):
        print("FakeKernel.get_local_size()")
        return FakeKernel._execution_state._local_size[index]

    @staticmethod
    def local_array(shape, dtype):
        print("FakeKernel.local_array()")
        current = FakeKernel._execution_state._current_local_array[0]
        if FakeKernel._execution_state._current_task[0] == 0:
            arr = np.zeros(shape, dtype)
            FakeKernel._execution_state._local_arrays.append(arr)
        else:
            arr = FakeKernel._execution_state._local_arrays[current]
        FakeKernel._execution_state._current_local_array[0] = current + 1
        return arr

    @staticmethod
    def private_array(shape, dtype):
        print("FakeKernel.private_array()")
        return np.zeros(shape, dtype)

    @staticmethod
    def group_reduce(value, op):
        print("FakeKernel.group_reduce()")
        return FakeKernel._execution_state._reduce(value, op)

    @staticmethod
    def barrier(flags):
        print("FakeKernel.barrier()")
        FakeKernel._execution_state._barrier()

    @staticmethod
    def mem_fence(flags):
        print("FakeKernel.mem_fence()")
        pass  # Nothing

    class atomic:
        @staticmethod
        def add(arr, ind, val):
            print("FakeKernel.atomic.add()")
            new_val = arr[ind] + val
            arr[ind] = new_val
            return new_val

        @staticmethod
        def sub(arr, ind, val):
            print("FakeKernel.atomic.sub()")
            new_val = arr[ind] - val
            arr[ind] = new_val
            return new_val

    class local:
        @staticmethod
        def array(shape, dtype):
            print("FakeKernel.local.array()")
            return FakeKernel.local_array(shape, dtype)

    class private:
        @staticmethod
        def array(shape, dtype):
            print("FakeKernel.private.array()")
            return FakeKernel.private_array(shape, dtype)

    class group:
        @staticmethod
        def reduce_add(value):
            print("FakeKernel.group.reduce_add()")
            return FakeKernel.group_reduce(value, lambda a, b: a + b)
