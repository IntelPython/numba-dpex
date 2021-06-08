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
import sys


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)   # numba-kernel-breakpoint
    l1 = a[i]                   # second-line
    l2 = b[i]                   # third-line
    c[i] = l1 + l2              # fourth-line


def driver(args):
    a = args[0]
    b = args[1]
    c = args[2]
    global_size = args[3]
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)


def get_context():
    if len(sys.argv) == 1 or sys.argv[1] == "gpu":
        device = dpctl.select_gpu_device()
    elif sys.argv[1] == "cpu":
        device = dpctl.select_cpu_device()
    else:
        raise Exception("Device doesn't supported.")
    print("Scheduling on ...")
    device.print_device_info()
    return device


def main():
    global_size = 10
    N = global_size

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    # Schedule on the queue requested at the command line.
    args = [a, b, c, global_size]

    context = get_context()

    with dpctl.device_context(context):
        driver(args)

    print("Done...")


if __name__ == "__main__":
    main()
