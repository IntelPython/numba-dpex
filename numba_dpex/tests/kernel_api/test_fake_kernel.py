# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations

import dpctl
import dpnp
import numpy as np
import pytest
from numpy.testing import assert_equal

import numba_dpex as ndpx
import numba_dpex.experimental as ndpx_ex
from numba_dpex.tests._helper import filter_strings

global_size = 10
local_size = 1
N = global_size * local_size


@pytest.fixture(params=[dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64])
def input_arrays(request):
    # The size of input and out arrays to be used
    a = dpnp.array(dpnp.random.random(N), request.param)
    b = dpnp.array(dpnp.random.random(N), request.param)
    c = dpnp.zeros_like(a)
    return a, b, c


def test_simple_sum3d():
    def sum3d(a, b, c):
        i = ndpx.get_global_id(0)
        j = ndpx.get_global_id(1)
        k = ndpx.get_global_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    a = dpnp.array([[[1, 2, 3], [4, 5, 6]]], dpnp.int64)
    b = dpnp.array([[[7, 8, 9], [10, 11, 12]]], dpnp.int64)
    c = dpnp.zeros(a.shape, dtype=a.dtype)
    c_sim = dpnp.zeros(a.shape, dtype=a.dtype)

    # Call sim kernel
    ndpx_ex.call_kernel(sum3d, ndpx.Range(*a.shape), a, b, c_sim)

    # Make kernel
    kfunc = ndpx_ex.kernel(sum3d)
    # Call kernel
    ndpx_ex.call_kernel(kfunc, ndpx.Range(*a.shape), a, b, c)

    c_exp = a + b

    assert_equal(c_sim.asnumpy(), c.asnumpy())
    assert_equal(c_sim.asnumpy(), c_exp.asnumpy())


@pytest.mark.parametrize("device_str", filter_strings)
def test_with_device_context(device_str, input_arrays):
    def sum2d(a, b, c):
        i = ndpx.get_global_id(0)
        c[i] = a[i] + b[i]

    a, b, c = input_arrays
    c_exp = a + b
    c_sim = dpnp.zeros_like(c)

    device = dpctl.SyclDevice(device_str)

    with dpctl.device_context(device):
        # Call dpex kernel
        ndpx_ex.call_kernel(
            ndpx_ex.kernel(sum2d),
            ndpx.NdRange(ndpx.Range(global_size), ndpx.Range(local_size)),
            a,
            b,
            c,
        )
        # Call sim kernel
        ndpx_ex.call_kernel(
            sum2d,
            ndpx.NdRange(ndpx.Range(global_size), ndpx.Range(local_size)),
            a,
            b,
            c_sim,
        )

    np.testing.assert_allclose(c.asnumpy(), c_exp.asnumpy(), rtol=1e-5, atol=0)
    np.testing.assert_allclose(
        c_sim.asnumpy(), c_exp.asnumpy(), rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(c_sim.asnumpy(), c.asnumpy(), rtol=1e-5, atol=0)


@pytest.mark.parametrize("device_str", filter_strings)
@pytest.mark.parametrize(
    "shape",
    [
        ndpx.Range(1),
        ndpx.Range(7),
        ndpx.Range(1, 1),
        ndpx.Range(7, 13),
        ndpx.Range(1, 1, 1),
        ndpx.Range(7, 13, 23),
    ],
)
def test_get_global_id(device_str, shape):
    def func1(c):
        i = ndpx.get_global_id(0)
        c[i] = i

    def func2(c):
        i = ndpx.get_global_id(0)
        j = ndpx.get_global_id(1)
        c[i, j] = i + j * 100

    def func3(c):
        i = ndpx.get_global_id(0)
        j = ndpx.get_global_id(1)
        k = ndpx.get_global_id(2)
        c[i, j, k] = i + j * 100 + k * 10000

    func = [func1, func2, func3][len(shape) - 1]
    kfunc = ndpx_ex.kernel(func)

    sim_res = dpnp.zeros(shape, dtype=dpnp.int32)
    ndpx_ex.call_kernel(func, shape, sim_res)

    gpu_res = dpnp.zeros(shape, dtype=dpnp.int32, device=device_str)
    ndpx_ex.call_kernel(kfunc, shape, gpu_res)

    assert_equal(gpu_res.asnumpy(), sim_res.asnumpy())


def _get_ndrange_permutations():
    dims = [6, 12, 24]

    result = []
    globals_sizes = (
        list(permutations(dims, 1))
        + list(permutations(dims, 2))
        + list(permutations(dims, 3))
    )
    for gz in globals_sizes:
        local_sizes = list(permutations([1, 3], len(gz)))
        for lz in local_sizes:
            if any([lv > gv for gv, lv in zip(gz, lz)]):
                continue

            result.append(ndpx.NdRange(gz, lz))

    return result


_ndranges = _get_ndrange_permutations()


@pytest.mark.parametrize("device_str", filter_strings)
@pytest.mark.parametrize("shape", _ndranges, ids=list(map(str, _ndranges)))
def test_get_local_id(device_str, shape):
    def func1(c):
        i = ndpx.get_global_id(0)
        li = ndpx.get_local_id(0)
        c[i] = li

    def func2(c):
        i = ndpx.get_global_id(0)
        j = ndpx.get_global_id(1)
        li = ndpx.get_local_id(0)
        lj = ndpx.get_local_id(1)
        c[i, j] = li + lj * 100

    def func3(c):
        i = ndpx.get_global_id(0)
        j = ndpx.get_global_id(1)
        k = ndpx.get_global_id(2)
        li = ndpx.get_local_id(0)
        lj = ndpx.get_local_id(1)
        lk = ndpx.get_local_id(2)
        c[i, j, k] = li + lj * 100 + lk * 10000

    global_size = shape.global_range

    func = [func1, func2, func3][len(global_size) - 1]
    kfunc = ndpx_ex.kernel(func)

    sim_res = dpnp.zeros(global_size, dtype=dpnp.int32)
    ndpx_ex.call_kernel(func, shape, sim_res)

    gpu_res = dpnp.zeros(global_size, dtype=dpnp.int32, device=device_str)
    ndpx_ex.call_kernel(kfunc, shape, gpu_res)

    assert_equal(gpu_res.asnumpy(), sim_res.asnumpy())
