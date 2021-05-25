#! /usr/bin/env python
# Copyright 2020, 2021 Intel Corporation
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

import dpctl
<<<<<<< HEAD
=======
import numpy.testing as testing
>>>>>>> main
import numba_dppy as dppy
import numpy as np

import dpctl.tensor as dpt


@dppy.kernel
def data_parallel_sum(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


def driver(a, b, c, global_size):
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)


def main():
    global_size = 10
    N = global_size
    print("N", N)

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    try:
        device = dpctl.select_default_device()
        print("Scheduling on ...")
        device.print_device_info()
        with dpctl.device_context(device):
            da = dpt.usm_ndarray(a.shape, dtype=a.dtype, buffer="shared")
            da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

            db = dpt.usm_ndarray(b.shape, dtype=b.dtype, buffer="shared")
            db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

            dc = dpt.usm_ndarray(c.shape, dtype=c.dtype, buffer="shared")

            driver(da, db, dc, global_size)

            res = dc.usm_data.copy_to_host().view(dc.dtype)
            c = np.ndarray(c.shape, buffer=res, dtype=c.dtype)
            assert np.array_equal(c, a + b)
    except ValueError:
        print("Failed to schedule on a SYCL device")

    print("Done...")


if __name__ == "__main__":
    main()
