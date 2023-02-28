# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import ir
from numba.core.ir_utils import find_topo_order

from numba_dpex.core.passes.parfor_lowering_pass import _lower_parfor_as_kernel
from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray

from .parfor import (
    Parfor,
    ParforPassStates,
    get_parfor_outputs,
    get_parfor_params,
)


class ParforLegalizeCFDPass(ParforPassStates):

    """ParforCFDPass class is responsible for enforcing CFD"""

    def run(self):
        # Get parfor params to calculate reductions below.
        _, parfors = get_parfor_params(
            self.func_ir.blocks,
            self.options.fusion,
            self.nested_fusion_info,
        )
        inputUsmTypeDict = {"device": 3, "shared": 2, "host": 1}

        paramsNameSet = set()
        from numba_dpex.core.exceptions import ComputeFollowsDataInferenceError

        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            block = self.func_ir.blocks[label]
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(
                    stmt.value, ir.Expr
                ):
                    # lhs = stmt.target.name
                    rhs = stmt.value
                    # check call, if there is dpnp array on the right, exit
                    if rhs.op == "call":
                        for ele in rhs.list_vars():
                            if ele.name in paramsNameSet:
                                raise ComputeFollowsDataInferenceError

                elif isinstance(stmt, Parfor):
                    if stmt.params is None:
                        continue

                    outputParams = get_parfor_outputs(stmt, stmt.params)
                    if len(outputParams) > 1:
                        raise AssertionError
                    inputDevice = None
                    inputUsmType = None
                    for para in stmt.params:
                        if not isinstance(self.typemap[para], DpnpNdArray):
                            continue
                        paramsNameSet.add(para)
                        # assuming DpnpNdArray either input or output, and no intermediate value
                        if para not in outputParams:
                            # checking device
                            if inputDevice is None:
                                inputDevice = self.typemap[para].device
                            if inputDevice != self.typemap[para].device:
                                raise ComputeFollowsDataInferenceError
                            usm_type = self.typemap[para].usm_type
                            # checking usm
                            if usm_type not in inputUsmTypeDict.keys():
                                raise ComputeFollowsDataInferenceError
                            if inputUsmType is None:
                                inputUsmType = usm_type
                            if (
                                inputUsmTypeDict[usm_type]
                                > inputUsmTypeDict[inputUsmType]
                            ):
                                inputUsmType = usm_type

                    # only update device and usm_type
                    # FIXME: sycl_queue will be updated later
                    if inputDevice is None:
                        raise AssertionError
                    for para in outputParams:
                        if not isinstance(self.typemap[para], DpnpNdArray):
                            continue
                        if (
                            self.typemap[para].device == inputDevice
                            and self.typemap[para].usm_type == inputUsmType
                        ):
                            continue
                        # now update typemap
                        self.typemap[para].device = inputDevice
                        self.typemap[para].usm_type = inputUsmType
                    stmt.lowerer = _lower_parfor_as_kernel
