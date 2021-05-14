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

import numpy as np
import numba_dppy as dppy
import dpctl


def main():
    """
    The example demonstrates the use of numba_dppy's ``atomic_add`` intrinsic
    function on a SYCL GPU device. The ``dpctl.select_gpu_device`` is
    equivalent to ``sycl::gpu_selector`` and returns a sycl::device of type GPU.
    """

    @dppy.kernel
    def atomic_add(a):
        dppy.atomic.add(a, 0, 1)

    global_size = 100
    a = np.array([0])

    try:
        d = dpctl.select_gpu_device()
        with dpctl.device_context(d):
            print("Offloading to ...")
            d.print_device_info()
            atomic_add[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
            # Expected 100, because global_size = 100
            print(a)
    except ValueError:
        print("No SYCL GPU found.")


if __name__ == "__main__":
    main()
