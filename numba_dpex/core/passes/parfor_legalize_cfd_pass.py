# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import ir
from numba.core.ir_utils import find_topo_order

from numba_dpex.core.exceptions import ComputeFollowsDataInferenceError
from numba_dpex.core.passes.parfor_lowering_pass import _lower_parfor_as_kernel
from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray

from .parfor import (
    Parfor,
    ParforPassStates,
    get_parfor_outputs,
    get_parfor_params,
)


class ParforLegalizeCFDPass(ParforPassStates):
    """Legalizes the compute-follows-data based device attribute for parfor
    nodes.

    DpnpNdArray array-expressions populate the type of the left-hand-side (LHS)
    of each expression as a default DpnpNdArray instance derived from the
    __array_ufunc__ method of DpnpNdArray class. The pass fixes the LHS type by
    properly applying compute follows data programming model. The pass first
    checks if the right-hand-side (RHS) DpnpNdArray arguments are on the same
    device, else raising a ComputeFollowsDataInferenceError. Once the RHS has
    been validated, the LHS type is updated.

    The pass also updated the usm_type of the LHS based on a USM type
    propagation rule: device > shared > host. Thus, if the usm_type attribute of
    the RHS arrays are "device" and "shared" respectively, the LHS array's
    usm_type attribute will be "device".

    Once the pass has identified a parfor with DpnpNdArrays and legalized it,
    the "lowerer" attribute of the parfor is set to
    ``numba_dpex.core.passes.parfor_lowering_pass._lower_parfor_as_kernel`` so
    that the parfor node is lowered using Dpex's lowerer.

    """

    def run(self):
        # The get_parfor_params needs to be run here to initialize the parfor
        # nodes prior to using them.
        _, parfors = get_parfor_params(
            self.func_ir.blocks,
            self.options.fusion,
            self.nested_fusion_info,
        )
        inputUsmTypeDict = {"device": 3, "shared": 2, "host": 1}
        paramsNameSet = set()

        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            block = self.func_ir.blocks[label]
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(
                    stmt.value, ir.Expr
                ):
                    # lhs = stmt.target.name
                    rhs = stmt.value
                    # FIXME: Calls with DpnpNdArray arguments are not yet
                    # evaluated and legalized. Raise a NotImplementedError.
                    if rhs.op == "call":
                        for ele in rhs.list_vars():
                            if ele.name in paramsNameSet:
                                raise NotImplementedError(
                                    "Compute follows data is not currently "
                                    "supported for function calls."
                                )
                elif isinstance(stmt, Parfor):
                    if stmt.params is None:
                        continue
                    outputParams = get_parfor_outputs(stmt, stmt.params)
                    if len(outputParams) > 1:
                        raise AssertionError

                    rhsDeviceTypes = set()
                    rhsUsmTypes = []
                    rhsDpnpNdArrayArgnums = []

                    for param_num, para in enumerate(stmt.params):
                        if not isinstance(self.typemap[para], DpnpNdArray):
                            continue
                        paramsNameSet.add(para)
                        # assuming DpnpNdArray either input or output, and no
                        # intermediate value
                        if para not in outputParams:
                            argty = self.typemap[para]
                            rhsDpnpNdArrayArgnums.append(param_num)
                            rhsDeviceTypes.add(argty.device)
                            try:
                                rhsUsmTypes.append(
                                    inputUsmTypeDict[argty.usm_type]
                                )
                            except KeyError:
                                raise ValueError(
                                    "Unknown USM type encountered. Supported "
                                    "usm types are: device, shared and host."
                                )

                    # Check compute follows data on RHS dpnp array args
                    if len(rhsDeviceTypes) > 1:
                        raise ComputeFollowsDataInferenceError(
                            kernel_name=stmt.loc.short(),
                            usmarray_argnum_list=rhsDpnpNdArrayArgnums,
                        )

                    # Derive the RHS usm_type based on usm allocator
                    # precedence rule: device > shared > host
                    rhs_usm_ty = max(rhsUsmTypes)

                    # only update device and usm_type
                    # FIXME: sycl_queue will be updated later
                    if len(rhsDeviceTypes) == 0:
                        raise AssertionError

                    rhs_device_ty = rhsDeviceTypes.pop()

                    for para in outputParams:
                        if not isinstance(self.typemap[para], DpnpNdArray):
                            continue
                        # now update typemap of RHS
                        self.typemap[para].device = rhs_device_ty
                        self.typemap[para].usm_type = rhs_usm_ty

                    # We have legalized the parfor for CFD, so set the lowerer
                    # to dpex's lowerer.
                    stmt.lowerer = _lower_parfor_as_kernel
