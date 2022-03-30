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

from numba.core import compiler, dispatcher
from numba.core.registry import cpu_target
from numba.core.target_extension import dispatcher_registry, target_registry

import numba_dppy.config as dppy_config
from numba_dppy.target import DPEX_TARGET_NAME


class OffloadDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target

    def __init__(
        self,
        py_func,
        locals={},
        targetoptions={},
        impl_kind="direct",
        pipeline_class=compiler.Compiler,
    ):
        if dppy_config.HAS_NON_HOST_DEVICE:
            from numba_dppy.compiler import Compiler

            targetoptions["parallel"] = True
            dispatcher.Dispatcher.__init__(
                self,
                py_func,
                locals=locals,
                targetoptions=targetoptions,
                impl_kind=impl_kind,
                pipeline_class=Compiler,
            )
        else:
            print(
                "--------------------------------------------------------------"
            )
            print(
                "WARNING : DPEX pipeline ignored. Ensure drivers are installed."
            )
            print(
                "--------------------------------------------------------------"
            )
            dispatcher.Dispatcher.__init__(
                self,
                py_func,
                locals=locals,
                targetoptions=targetoptions,
                impl_kind=impl_kind,
                pipeline_class=pipeline_class,
            )


dispatcher_registry[target_registry[DPEX_TARGET_NAME]] = OffloadDispatcher
