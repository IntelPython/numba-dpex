# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A module for kernel simulator

This module implements a kernel simulator. The simulator will be mostly used
for debugging basic python functions before compiling as numba_dpex kernels.
The kernel simulator just runs a python function instead of compiling it.
The execution of the function can be serial or parallel. Either case, the
simulator code will "fake" different sycl kernel parameters and functions, also
how they interact with each other, e.g. `get_global_id()`, `barrier`,
`local.array` etc. This module is composed of two classes -- `ExecutionState`
and `FakeKernel`.
"""

from functools import reduce
from itertools import product

import numpy as np

try:
    from greenlet import greenlet

    _GREENLET_FOUND = True
except ImportError:
    _GREENLET_FOUND = False

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
    """Class to simulate a kernel execution state"""

    def __init__(self, global_size):
        self.indices = [0] * len(global_size)
        self.wg_size = [None]
        self.tasks = []
        self.current_task = [None]
        self.local_array = []
        self.current_local_array = [0]
        self.reduce_val = [None]

    def reset(self, wg_size):
        """_summary_

        Args:
            wg_size (_type_): _description_
        """
        self.wg_size[0] = wg_size
        self.current_task[0] = 0
        self.local_array.clear()
        self.current_local_array[0] = 0

    def barrier(self):
        """_summary_"""
        assert self.wg_size > 0
        if self.wg_size > 1:
            assert len(self.tasks) > 0
            next_task = self.current_task[0] + 1
            if next_task >= self.wg_size:
                next_task = 0
            self.current_task[0] = next_task
            self.tasks[next_task].switch()

    def reduce(self, value, operator):
        """_summary_

        Args:
            value (_type_): _description_
            operator (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.current_task[0] == 0:
            self.reduce_val[0] = value
        else:
            self.reduce_val[0] = operator(self.reduce_val[0], value)
        self.barrier()
        return self.reduce_val[0]

    def capture_func(self, func, indices, args):
        """_summary_

        Args:
            func (_type_): _description_
            indices (_type_): _description_
            args (_type_): _description_
        """

        def wrapper():
            self.indices[:] = indices
            func(*args)

        return wrapper


# TODO: Share code with dispatcher
class FakeKernel:
    """Class for a kernel simulator.

    Raises:
        UnsupportedKernelArgumentError: _description_

    Returns:
        _type_: _description_
    """

    execution_state = None

    def __init__(
        self,
        func,
        index_space,
        *kernel_args,
    ):
        self._func = func
        FakeKernel.global_range, FakeKernel.local_range = self._get_ranges(
            index_space
        )
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
            # TODO: fix group and group.reduce_add
            # (group, group_proxy), # noqa: E800
            # (group.reduce_add, group_proxy.reduce_add),   # noqa: E800
        ]

        self._need_barrier = self._has_barrier_ops()
        FakeKernel.execution_state = ExecutionState(FakeKernel.global_range)

        self._saved_globals = self._replace_globals(self._func.__globals__)
        self._saved_closure = self._replace_closure(self._func.__closure__)

    def _get_ranges(self, index_space):
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
        for val in self._func.__globals__.values():
            if val is barrier:
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
        if self._saved_globals is None:
            return
        src.update(self._saved_globals)

    def _replace_closure(self, src):
        if src is None:
            return None

        old_vals = [elem.cell_contents for elem in src]
        for elem in src:
            old_val = elem.cell_contents
            elem.cell_contents = self._replace_global_func(old_val)
        return old_vals

    def _restore_closure(self, src):
        if self._saved_closure is None:
            return

        for i in range(len(src)):  # pylint: disable=consider-using-enumerate
            src[i].cell_contents = self._saved_closure[i]

    def execute(self):
        """Execute a simulated kernel"""
        assert FakeKernel.execution_state
        try:
            groups = tuple(
                (g + l - 1) // l
                for g, l in zip(FakeKernel.global_range, FakeKernel.local_range)
            )
            for gid in product(*(range(g) for g in groups)):
                offset = tuple(
                    g * l for g, l in zip(gid, FakeKernel.local_range)
                )
                size = tuple(
                    min(g - o, l)
                    for o, g, l in zip(
                        offset, FakeKernel.global_range, FakeKernel.local_range
                    )
                )
                count = reduce(lambda a, b: a * b, size)
                FakeKernel.execution_state.reset(count)

                indices_range = (range(o, o + s) for o, s in zip(offset, size))

                if self._need_barrier:
                    # global _GREENLET_FOUND
                    assert _GREENLET_FOUND, "greenlet package not installed"
                    assert len(FakeKernel.execution_state.tasks) == 0
                    for indices in product(*indices_range):
                        FakeKernel.execution_state.tasks.append(
                            greenlet(
                                FakeKernel.execution_state.capture_func(
                                    self._func, indices, self._kernel_args
                                )
                            )
                        )

                    for tsk in FakeKernel.execution_state.tasks:
                        tsk.switch()

                    FakeKernel.execution_state.tasks.clear()
                else:
                    for indices in product(*indices_range):
                        FakeKernel.execution_state.indices[:] = indices
                        self._func(*self._kernel_args)

        finally:
            self._restore_closure(self._func.__closure__)
            self._restore_globals(self._func.__globals__)
            FakeKernel.execution_state = None

    @staticmethod
    def get_global_id(index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        return FakeKernel.execution_state.indices[index]

    @staticmethod
    def get_local_id(index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            FakeKernel.execution_state.indices[index]
            % FakeKernel.local_range[index]
        )

    @staticmethod
    def get_group_id(index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            FakeKernel.execution_state.indices[index]
            // FakeKernel.local_range[index]
        )

    @staticmethod
    def get_global_size(index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        # pylint: disable-next=unsubscriptable-object
        return FakeKernel.global_range[index]

    @staticmethod
    def get_local_size(index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        # pylint: disable-next=unsubscriptable-object
        return FakeKernel.local_range[index]

    @staticmethod
    def local_array(shape, dtype):
        """_summary_

        Args:
            shape (_type_): _description_
            dtype (_type_): _description_

        Returns:
            _type_: _description_
        """
        current = FakeKernel.execution_state.current_local_array[0]
        if FakeKernel.execution_state.current_task[0] == 0:
            arr = np.zeros(shape, dtype)
            FakeKernel.execution_state.local_array.append(arr)
        else:
            arr = FakeKernel.execution_state.local_array[current]
        FakeKernel.execution_state.current_local_array[0] = current + 1
        return arr

    @staticmethod
    def private_array(shape, dtype):
        """_summary_

        Args:
            shape (_type_): _description_
            dtype (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.zeros(shape, dtype)

    @staticmethod
    def group_reduce(value, operator):
        """_summary_

        Args:
            value (_type_): _description_
            operator (_type_): _description_

        Returns:
            _type_: _description_
        """
        return FakeKernel.execution_state.reduce(value, operator)

    @staticmethod
    def barrier(flags):  # pylint: disable=unused-argument
        """_summary_

        Args:
            flags (_type_): _description_
        """
        FakeKernel.execution_state.barrier()

    @staticmethod
    def mem_fence(flags):  # pylint: disable=unused-argument
        """_summary_

        Args:
            flags (_type_): _description_
        """
        pass  # pylint: disable=unnecessary-pass

    class atomic:  # pylint: disable=invalid-name
        """_summary_

        Returns:
            _type_: _description_
        """

        @staticmethod
        def add(arr, ind, val):
            """_summary_

            Args:
                arr (_type_): _description_
                ind (_type_): _description_
                val (_type_): _description_

            Returns:
                _type_: _description_
            """
            new_val = arr[ind] + val
            arr[ind] = new_val
            return new_val

        @staticmethod
        def sub(arr, ind, val):
            """_summary_

            Args:
                arr (_type_): _description_
                ind (_type_): _description_
                val (_type_): _description_

            Returns:
                _type_: _description_
            """
            new_val = arr[ind] - val
            arr[ind] = new_val
            return new_val

    class local:  # pylint: disable=invalid-name, too-few-public-methods
        """_summary_

        Returns:
            _type_: _description_
        """

        @staticmethod
        def array(shape, dtype):
            """_summary_

            Args:
                shape (_type_): _description_
                dtype (_type_): _description_

            Returns:
                _type_: _description_
            """
            return FakeKernel.local_array(shape, dtype)

    class private:  # pylint: disable=invalid-name, too-few-public-methods
        """_summary_

        Returns:
            _type_: _description_
        """

        @staticmethod
        def array(shape, dtype):
            """_summary_

            Args:
                shape (_type_): _description_
                dtype (_type_): _description_

            Returns:
                _type_: _description_
            """
            return FakeKernel.private_array(shape, dtype)

    class group:  # pylint: disable=invalid-name, too-few-public-methods
        """_summary_

        Returns:
            _type_: _description_
        """

        @staticmethod
        def reduce_add(value):
            """_summary_

            Args:
                value (_type_): _description_

            Returns:
                _type_: _description_
            """
            return FakeKernel.group_reduce(value, lambda a, b: a + b)
