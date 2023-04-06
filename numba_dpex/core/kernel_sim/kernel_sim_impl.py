# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from collections import namedtuple
from functools import reduce
from itertools import product

import numpy as np

try:
    from greenlet import greenlet

    _greenlet_found = True
except ImportError:
    _greenlet_found = False


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

_execution_state = None


def get_exec_state():
    global _execution_state
    assert _execution_state is not None
    return _execution_state


def _save_local_state(state):
    indices = copy.deepcopy(state.indices)
    current_local_array = state.current_local_array[0]
    state.current_local_array[0] = 0
    return (indices, current_local_array)


def _restore_local_state(state, saved_state):
    state.indices[:] = saved_state[0]
    state.current_local_array[0] = saved_state[1]


def _reset_local_state(state, wg_size):
    state.wg_size[0] = wg_size
    state.current_task[0] = 0
    state.local_arrays.clear()
    state.current_local_array[0] = 0


def _barrier_impl(state):
    wg_size = state.wg_size[0]
    assert wg_size > 0
    if wg_size > 1:
        assert len(state.tasks) > 0
        saved_state = _save_local_state(state)
        next_task = state.current_task[0] + 1
        if next_task >= wg_size:
            next_task = 0
        state.current_task[0] = next_task
        state.tasks[next_task].switch()
        _restore_local_state(state, saved_state)


def _reduce_impl(state, value, op):
    if state.current_task[0] == 0:
        state.reduce_val[0] = value
    else:
        state.reduce_val[0] = op(state.reduce_val[0], value)
    _barrier_impl(state)
    return state.reduce_val[0]


def _setup_execution_state(global_size, local_size):
    global _execution_state
    assert _execution_state is None

    _execution_state = _ExecutionState(
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
    return _execution_state


def _destroy_execution_state():
    global _execution_state
    _execution_state = None


def _replace_globals(src, replace_global_func):
    old_globals = list(src.items())
    for name, val in src.items():
        src[name] = replace_global_func(val)

    return old_globals


def _restore_globals(src, old_globals):
    src.update(old_globals)


def _replace_closure(src, replace_global_func):
    if src is None:
        return None

    old_vals = [e.cell_contents for e in src]
    for e in src:
        old_val = e.cell_contents
        e.cell_contents = replace_global_func(old_val)
    return old_vals


def _restore_closure(src, old_closure):
    if old_closure is None:
        return

    for i in range(len(src)):
        src[i].cell_contents = old_closure[i]


def _capture_func(func, indices, args):
    def wrapper():
        get_exec_state().indices[:] = indices
        func(*args)

    return wrapper


def barrier():
    _barrier_impl(get_exec_state())


def group_reduce(value, op):
    return _reduce_impl(get_exec_state(), value, op)


def local_array(shape, dtype):
    state = get_exec_state()
    current = state.current_local_array[0]
    if state.current_task[0] == 0:
        arr = np.zeros(shape, dtype)
        state.local_arrays.append(arr)
    else:
        arr = state.local_arrays[current]
    state.current_local_array[0] = current + 1
    return arr


def private_array(shape, dtype):
    return np.zeros(shape, dtype)


def get_global_id(index):
    return get_exec_state().indices[index]


def get_local_id(index):
    state = get_exec_state()
    return state.indices[index] % state.local_size[index]


def get_group_id(index):
    state = get_exec_state()
    return state.indices[index] // state.local_size[index]


def get_global_size(index):
    return get_exec_state().global_size[index]


def get_local_size(index):
    return get_exec_state().local_size[index]


def execute_kernel(
    global_size, local_size, func, args, need_barrier, replace_global_func
):
    if local_size is None or len(local_size) == 0:
        local_size = (1,) * len(global_size)

    saved_globals = _replace_globals(func.__globals__, replace_global_func)
    saved_closure = _replace_closure(func.__closure__, replace_global_func)
    state = _setup_execution_state(global_size, local_size)
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
            _reset_local_state(state, count)

            indices_range = (range(o, o + s) for o, s in zip(offset, size))

            if need_barrier:
                global _greenlet_found
                assert _greenlet_found, "greenlet package not installed"
                tasks = state.tasks
                assert len(tasks) == 0
                for indices in product(*indices_range):
                    tasks.append(greenlet(_capture_func(func, indices, args)))

                for t in tasks:
                    t.switch()

                tasks.clear()
            else:
                for indices in product(*indices_range):
                    state.indices[:] = indices
                    func(*args)

    finally:
        _restore_closure(func.__closure__, saved_closure)
        _restore_globals(func.__globals__, saved_globals)
        _destroy_execution_state()
