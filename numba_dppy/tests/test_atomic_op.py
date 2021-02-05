# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import numba
import numba_dppy, numba_dppy as dppy
import unittest
import dpctl


def atomic_add_int32(ary):
    tid = dppy.get_local_id(0)
    lm = dppy.local.static_alloc(32, numba.uint32)
    lm[tid] = 0
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    dppy.atomic.add(lm, bin, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = lm[tid]


def atomic_sub_int32(ary):
    tid = dppy.get_local_id(0)
    lm = dppy.local.static_alloc(32, numba.uint32)
    lm[tid] = 0
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    dppy.atomic.sub(lm, bin, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = lm[tid]


def atomic_add_float32(ary):
    lm = dppy.local.static_alloc(1, numba.float32)
    lm[0] = ary[0]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.add(lm, 0, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_sub_float32(ary):
    lm = dppy.local.static_alloc(1, numba.float32)
    lm[0] = ary[0]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.sub(lm, 0, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_add_int64(ary):
    lm = dppy.local.static_alloc(1, numba.int64)
    lm[0] = ary[0]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.add(lm, 0, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_sub_int64(ary):
    lm = dppy.local.static_alloc(1, numba.int64)
    lm[0] = ary[0]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.sub(lm, 0, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_add_float64(ary):
    lm = dppy.local.static_alloc(1, numba.float64)
    lm[0] = ary[0]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.add(lm, 0, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_sub_float64(ary):
    lm = dppy.local.static_alloc(1, numba.float64)
    lm[0] = ary[0]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.sub(lm, 0, 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_add2(ary):
    tx = dppy.get_local_id(0)
    ty = dppy.get_local_id(1)
    lm = dppy.local.static_alloc((4, 8), numba.uint32)
    lm[tx, ty] = ary[tx, ty]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.add(lm, (tx, ty), 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = lm[tx, ty]


def atomic_add3(ary):
    tx = dppy.get_local_id(0)
    ty = dppy.get_local_id(1)
    lm = dppy.local.static_alloc((4, 8), numba.uint32)
    lm[tx, ty] = ary[tx, ty]
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    dppy.atomic.add(lm, (tx, numba.uint64(ty)), 1)
    dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = lm[tx, ty]


def call_fn_for_datatypes(fn, result, input, global_size):
    dtypes = [np.int32, np.int64, np.float32, np.double]

    for dtype in dtypes:
        a = np.array(input, dtype=dtype)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            # if dtype == np.double and not device_env.device_support_float64_atomics():
            #    continue
            # if dtype == np.int64 and not device_env.device_support_int64_atomics():
            #    continue
            fn[global_size, dppy.DEFAULT_LOCAL_SIZE](a)

        assert a[0] == result


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
@unittest.skipUnless(
    numba_dppy.ocl.atomic_support_present(), "test only when atomic support is present"
)
class TestAtomicOp(unittest.TestCase):
    def test_atomic_add_global(self):
        @dppy.kernel
        def atomic_add(B):
            dppy.atomic.add(B, 0, 1)

        N = 100
        B = np.array([0])

        call_fn_for_datatypes(atomic_add, N, B, N)

    def test_atomic_sub_global(self):
        @dppy.kernel
        def atomic_sub(B):
            dppy.atomic.sub(B, 0, 1)

        N = 100
        B = np.array([100])

        call_fn_for_datatypes(atomic_sub, 0, B, N)

    def test_atomic_add_local_int32(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()

        # dppy_atomic_add = dppy.kernel('void(uint32[:])')(atomic_add_int32)
        dppy_atomic_add = dppy.kernel(atomic_add_int32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppy_atomic_add[32, dppy.DEFAULT_LOCAL_SIZE](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_sub_local_int32(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()

        # dppy_atomic_sub = dppy.kernel('void(uint32[:])')(atomic_sub_int32)
        dppy_atomic_sub = dppy.kernel(atomic_sub_int32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppy_atomic_sub[32, dppy.DEFAULT_LOCAL_SIZE](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] -= 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add_local_float32(self):
        ary = np.array([0], dtype=np.float32)

        # dppy_atomic_add = dppy.kernel('void(float32[:])')(atomic_add_float32)
        dppy_atomic_add = dppy.kernel(atomic_add_float32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppy_atomic_add[32, dppy.DEFAULT_LOCAL_SIZE](ary)

        self.assertTrue(ary[0] == 32)

    def test_atomic_sub_local_float32(self):
        ary = np.array([32], dtype=np.float32)

        # dppy_atomic_sub = dppy.kernel('void(float32[:])')(atomic_sub_float32)
        dppy_atomic_sub = dppy.kernel(atomic_sub_float32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:

            dppy_atomic_sub[32, dppy.DEFAULT_LOCAL_SIZE](ary)

        self.assertTrue(ary[0] == 0)

    def test_atomic_add_local_int64(self):
        ary = np.array([0], dtype=np.int64)

        # dppy_atomic_add = dppy.kernel('void(int64[:])')(atomic_add_int64)
        dppy_atomic_add = dppy.kernel(atomic_add_int64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            # if device_env.device_support_int64_atomics():
            dppy_atomic_add[32, dppy.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 32)
            # else:
            #    return

    def test_atomic_sub_local_int64(self):
        ary = np.array([32], dtype=np.int64)

        # fn = dppy.kernel('void(int64[:])')(atomic_sub_int64)
        fn = dppy.kernel(atomic_sub_int64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            # if device_env.device_support_int64_atomics():
            fn[32, dppy.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 0)
            # else:
            #    return

    def test_atomic_add_local_float64(self):
        ary = np.array([0], dtype=np.double)

        # fn = dppy.kernel('void(float64[:])')(atomic_add_float64)
        fn = dppy.kernel(atomic_add_float64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            # if device_env.device_support_float64_atomics():
            fn[32, dppy.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 32)
            # else:
            #    return

    def test_atomic_sub_local_float64(self):
        ary = np.array([32], dtype=np.double)

        # fn = dppy.kernel('void(float64[:])')(atomic_sub_int64)
        fn = dppy.kernel(atomic_sub_int64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            # if device_env.device_support_float64_atomics():
            fn[32, dppy.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 0)
            # else:
            #    return

    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        # dppy_atomic_add2 = dppy.kernel('void(uint32[:,:])')(atomic_add2)
        dppy_atomic_add2 = dppy.kernel(atomic_add2)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppy_atomic_add2[(4, 8), dppy.DEFAULT_LOCAL_SIZE](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        # dppy_atomic_add3 = dppy.kernel('void(uint32[:,:])')(atomic_add3)
        dppy_atomic_add3 = dppy.kernel(atomic_add3)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppy_atomic_add3[(4, 8), dppy.DEFAULT_LOCAL_SIZE](ary)

        self.assertTrue(np.all(ary == orig + 1))


if __name__ == "__main__":
    unittest.main()
