#! /usr/bin/env python
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

from . import _helper
import numpy as np
import numba_dppy as dppy
import dpctl


@dppy.kernel(
    access_types={"read_only": ["a", "b"], "write_only": ["c"], "read_write": []}
)
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
local_size = 32
N = global_size * local_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)


def main():
    if _helper.has_gpu():
        with dpctl.device_context("opencl:gpu") as queue:
            print("Offloading to ...")
            queue.get_sycl_device().print_device_info()
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, local_size](a, b, c)
            print("after  C: ", c)
    else:
        print("Could not find an OpenCL GPU device")

    if _helper.has_cpu():
        with dpctl.device_context("opencl:cpu") as queue:
            print("Offloading to ...")
            queue.get_sycl_device().print_device_info()
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, local_size](a, b, c)
            print("after  C: ", c)
    else:
        print("Could not find an OpenCL CPU device")

    if _helper.has_gpu("level_zero"):
        with dpctl.device_context("level_zero:gpu") as queue:
            print("Offloading to ...")
            queue.get_sycl_device().print_device_info()
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, local_size](a, b, c)
            print("after  C: ", c)
    else:
        print("Could not find an Level Zero GPU device")
    print("Done...")


if __name__ == "__main__":
    main()
