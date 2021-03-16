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

from __future__ import absolute_import, print_function
import llvmlite.binding as ll
import os
from ctypes.util import find_library
from numba_dppy.vectorizers import DPPYVectorize
from numba.np.ufunc.decorators import Vectorize


def init_jit():
    from numba_dppy.dispatcher import DPPYDispatcher

    return DPPYDispatcher


def initialize_all():
    from numba.core.registry import dispatcher_registry

    dispatcher_registry.ondemand["dppy"] = init_jit

    import dpctl
    import glob
    import platform as plt

    platform = plt.system()
    if platform == "Windows":
        paths = glob.glob(
            os.path.join(os.path.dirname(dpctl.__file__), "*DPCTLSyclInterface.dll")
        )
    else:
        paths = glob.glob(
            os.path.join(os.path.dirname(dpctl.__file__), "*DPCTLSyclInterface*")
        )

    if len(paths) == 1:
        ll.load_library_permanently(paths[0])
    else:
        raise ImportError

    ll.load_library_permanently(find_library("OpenCL"))

    def init_dppy_vectorize():
        return DPPYVectorize

    Vectorize.target_registry.ondemand["dppy"] = init_dppy_vectorize
