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

from numba.core import dispatcher, compiler
from numba.core.registry import cpu_target, dispatcher_registry
from numba_dppy import config


class DppyOffloadDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target

    def __init__(
        self,
        py_func,
        locals={},
        targetoptions={},
        impl_kind="direct",
        pipeline_class=compiler.Compiler,
    ):
        if config.dppy_present:
            from numba_dppy.compiler import DPPYCompiler

            targetoptions["parallel"] = True
            dispatcher.Dispatcher.__init__(
                self,
                py_func,
                locals=locals,
                targetoptions=targetoptions,
                impl_kind=impl_kind,
                pipeline_class=DPPYCompiler,
            )
        else:
            print(
                "---------------------------------------------------------------------"
            )
            print(
                "WARNING : DPPY pipeline ignored. Ensure OpenCL drivers are installed."
            )
            print(
                "---------------------------------------------------------------------"
            )
            dispatcher.Dispatcher.__init__(
                self,
                py_func,
                locals=locals,
                targetoptions=targetoptions,
                impl_kind=impl_kind,
                pipeline_class=pipeline_class,
            )


dispatcher_registry["__dppy_offload_gpu__"] = DppyOffloadDispatcher
dispatcher_registry["__dppy_offload_cpu__"] = DppyOffloadDispatcher
