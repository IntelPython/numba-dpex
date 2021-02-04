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

"""
from numba.testing import SerialSuite
from numba.testing import load_testsuite
from os.path import dirname, join
"""

import numba_dppy
import numba_dppy.config as dppy_config

# from numba_dppy.tests.dppy import *

"""
def load_tests(loader, tests, pattern):

    suite = SerialSuite()

    if dppy_config.dppy_present:
        suite.addTests(load_testsuite(loader, dirname(__file__)))
    else:
        print("skipped DPPY tests")

    return suite
"""
