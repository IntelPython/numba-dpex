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
import sys
import numpy as np
import numpy.testing as testing
import numba_dppy as dppy
import dpctl


# Sample Kernel that performs simple element-wise add
@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


def driver(a, b, c, global_size):
    # Get dppy device arrays for input.
    # We can pass kw argument queue to `to_device`. This will
    # make sure the memory allocated for DeviceArray is
    # using the passed queue. In the case where the queue
    # is not passed the queue returned by `dpctl.get_current_queue`
    # will be used. The same queue argument can be passed when creating
    # DeviceArray as well.
    da = dppy.to_device(a)
    db = dppy.to_device(b)

    # Array `c` is write only. We can use the `to_device`
    # convenience function or create a DeviceArray.
    # Using the convenience function will copy the data of
    # the np.ndarray we pass as argument, which is redundant.
    dc = dppy.DeviceArray(a.shape, a.strides, a.dtype)

    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](da, db, dc)

    # copy_to_host will create a ndarray and copy the data from
    # the DeviceArray.
    d = dc.copy_to_host()
    assert np.allclose(d, a + b)

    # We can also send a np.ndarray and copy the data of the
    # DeviceArray in that ndarray.
    dc.copy_to_host(c)
    assert np.allclose(c, a + b)

    # We can mix and match DeviceArray and np.ndarray
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](c, b, dc)
    e = dc.copy_to_host()
    assert np.allclose(e, c + b)


def main():
    global_size = 10
    N = global_size

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.empty_like(a)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            driver(a, b, c, global_size)

    print("Done!")


if __name__ == "__main__":
    main()
