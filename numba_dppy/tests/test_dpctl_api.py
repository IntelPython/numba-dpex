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

import unittest
import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestDPCTLAPI(unittest.TestCase):
    def test_dpctl_api(self):
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dpctl.dump()
            dpctl.get_current_queue()
            dpctl.get_num_platforms()
            dpctl.get_num_activated_queues()
            dpctl.has_cpu_queues()
            dpctl.has_gpu_queues()
            dpctl.has_sycl_platforms()
            dpctl.is_in_device_context()


if __name__ == "__main__":
    unittest.main()
