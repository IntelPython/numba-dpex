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


def ensure_dpnp(name):
    try:
        from . import dpnp_fptr_interface as dpnp_iface
    except ImportError:
        raise ImportError("dpnp is needed to call np.%s" % name)


def _init_dpnp():
    try:
        import os

        import dpnp

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.dirname(dpnp.__file__))
    except ImportError:
        pass


_init_dpnp()

DEBUG = None
