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

import os

import llvmlite.binding as ll
from numba.np.ufunc.decorators import Vectorize

from numba_dppy.vectorizers import DPPYVectorize


def init_jit():
    from numba_dppy.dispatcher import DPPYDispatcher

    return DPPYDispatcher


def load_dpctl_sycl_interface():
    """Permanently loads the ``DPCTLSyclInterface`` library provided by dpctl.

    The ``DPCTLSyclInterface`` library provides C wrappers over SYCL functions
    that are directly invoked from the LLVM modules generated by numba-dppy.
    We load the library once at the time of initialization using llvmlite's
    load_library_permanently function.

    Raises:
        ImportError: If the ``DPCTLSyclInterface`` library could not be loaded.
    """
    import glob
    import platform as plt

    import dpctl

    platform = plt.system()
    if platform == "Windows":
        paths = glob.glob(
            os.path.join(os.path.dirname(dpctl.__file__), "*DPCTLSyclInterface.dll")
        )
    else:
        paths = glob.glob(
            os.path.join(os.path.dirname(dpctl.__file__), "*DPCTLSyclInterface.so")
        )

    if len(paths) == 1:
        ll.load_library_permanently(paths[0])
    else:
        raise ImportError

    def init_dppy_vectorize():
        return DPPYVectorize

    Vectorize.target_registry.ondemand["dppy"] = init_dppy_vectorize
