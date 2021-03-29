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

import re
import unittest

import numba_dppy as dppy
import dpctl

from numba.tests.support import TestCase
from numba.core import types, compiler
from numba_dppy import compiler


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestDebugInfo(TestCase):
    """
    These tests only check the compiled assembly for debuginfo.
    """
    def get_kernel_asm(self, fn, sig, debug=False):
        kernel = compiler.compile_kernel(fn.sycl_queue, fn.py_func, sig, None, debug=debug)
        return kernel.assembly

    def make_check(self, fn, sig, expect):
        asm = self.get_kernel_asm(fn, sig=sig, debug=expect)

        # Checking whether debug symbols have been emmited to IR
        m = re.search(r"!dbg", asm, re.I)
        got = m is not None
        self.assertEqual(expect, got, msg='debug info not found in kernel:\n%s' % fn)

    def test_debuginfo_in_asm(self):
        """
        Check debug info is emitting to IR if debug parameter is set to True
        """

        debug_expect = True

        @dppy.kernel
        def foo(x):
            return x

        if dpctl.has_gpu_queues():
            with dpctl.device_context("opencl:gpu") as gpu_queue:
                sig = (types.int32,)
                self.make_check(foo, sig, debug_expect)

    def test_debuginfo_not_in_asm(self):
        """
        Check debug info is not emitting to IR if debug parameter is set to True
        """

        debug_expect = False

        @dppy.kernel
        def foo(x):
            return x

        if dpctl.has_gpu_queues():
            with dpctl.device_context("opencl:gpu") as gpu_queue:
                sig = (types.int32,)
                self.make_check(foo, sig, debug_expect)


if __name__ == '__main__':
    unittest.main()
