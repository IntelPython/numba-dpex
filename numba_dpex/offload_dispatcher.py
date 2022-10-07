# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import compiler, dispatcher
from numba.core.registry import cpu_target
from numba.core.target_extension import dispatcher_registry, target_registry

import numba_dpex.config as dpex_config
from numba_dpex.target import DPEX_TARGET_NAME


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
        if dpex_config.HAS_NON_HOST_DEVICE:
            from numba_dpex.compiler import Compiler

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
