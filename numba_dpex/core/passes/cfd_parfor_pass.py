# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import ir
from numba.core.ir_utils import find_topo_order

from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray

from .parfor import (
    Parfor,
    ParforPassStates,
    get_parfor_outputs,
    get_parfor_params,
)


class ParforCFDPass(ParforPassStates):

    """ParforCFDPass class is responsible for enforcing CFD"""

    def run(self):
        # Get parfor params to calculate reductions below.
        _, parfors = get_parfor_params(
            self.func_ir.blocks,
            self.options.fusion,
            self.nested_fusion_info,
        )

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
                    intputDevice = None
                    for para in stmt.params:
                        if not isinstance(self.typemap[para], DpnpNdArray):
                            continue
                        # breakpoint()
                        paramsNameSet.add(para)
                        # assuming DpnpNdArray either input or output, and no intermediate value
                        if para not in outputParams:
                            if intputDevice is None:
                                intputDevice = self.typemap[para].device
                            if intputDevice != self.typemap[para].device:
                                raise ComputeFollowsDataInferenceError

                    # only update device
                    # FIXME: sycl_queue will be updated later
                    if intputDevice is None:
                        raise AssertionError

                    if not isinstance(
                        self.typemap[outputParams[0]], DpnpNdArray
                    ):
                        continue
                    if self.typemap[outputParams[0]].device == intputDevice:
                        continue
                    # now update typemap
                    self.typemap[outputParams[0]].device = intputDevice
