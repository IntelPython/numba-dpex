# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy
import operator

import dpnp
import numba
from numba.core import ir, types
from numba.core.ir_utils import (
    get_np_ufunc_typ,
    legalize_names,
    remove_dels,
    replace_var_names,
)
from numba.parfors import parfor
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder

from numba_dpex.core.utils.cgutils_extra import get_llvm_type
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl

from ..types.dpnp_ndarray_type import DpnpNdArray


class ReductionHelper:
    """The class to define and allocate reduction intermediate variables."""

    def _allocate_partial_reduction_arrays(
        self,
        parfor,
        lowerer,
        red_name,
        inputArrayType,
    ):
        # reduction arrays outer dimension equal to thread count
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        pfbdr = ParforLoweringBuilder(lowerer=lowerer, scope=scope, loc=loc)

        # Get the type of the reduction variable.
        redvar_typ = lowerer.fndesc.typemap[red_name]

        # redarrvar_typ is type(partial_sum) # noqa: E800 help understanding
        redarrvar_typ = self._redtyp_to_redarraytype(redvar_typ, inputArrayType)
        reddtype = redarrvar_typ.dtype
        redarrdim = redarrvar_typ.ndim

        # allocate partial sum
        work_group_size = 8
        # writing work_group_size inot IR
        work_group_size_var = pfbdr.assign(
            rhs=ir.Const(work_group_size, loc),
            typ=types.literal(work_group_size),
            name="work_group_size",
        )

        # get total_work from parfor loop range
        # FIXME: right way is to use (stop - start) if start != 0
        total_work_var = pfbdr.assign(
            rhs=parfor.loop_nests[0].stop,
            typ=types.intp,
            name="tot_work",
        )

        # global_size_mod = tot_work%work_group_size # noqa: E800 help understanding
        ir_expr = ir.Expr.binop(
            operator.mod, total_work_var, work_group_size_var, loc
        )
        pfbdr._calltypes[ir_expr] = numba.core.typing.signature(
            types.intp, types.intp, types.intp
        )
        self.global_size_mod_var = pfbdr.assign(
            rhs=ir_expr, typ=types.intp, name="global_size_mod"
        )

        # Calculates global_size_var as total_work - global_size_mod
        ir_expr = ir.Expr.binop(
            operator.sub, total_work_var, self.global_size_mod_var, loc
        )
        pfbdr._calltypes[ir_expr] = numba.core.typing.signature(
            types.intp, types.intp, types.intp
        )
        self.global_size_var = pfbdr.assign(
            rhs=ir_expr, typ=types.intp, name="global_size"
        )

        # Calculates partial_sum_size_var as
        # global_size_var_assign // work_group_size_var_assign
        ir_expr = ir.Expr.binop(
            operator.floordiv,
            self.global_size_var,
            work_group_size_var,
            loc,
        )
        pfbdr._calltypes[ir_expr] = numba.core.typing.signature(
            types.intp, types.intp, types.intp
        )
        self.partial_sum_size_var = pfbdr.assign(
            rhs=ir_expr, typ=types.intp, name="partial_sum_size"
        )
        # Dpnp object
        fillFunc = None
        parfor_reddict = parfor.reddict
        redop = parfor_reddict[red_name].redop

        if redop == operator.iadd:
            fillFunc = dpnp.zeros
        elif redop == operator.imul:
            fillFunc = dpnp.ones
        else:
            raise NotImplementedError

        kws = {
            "shape": types.UniTuple(types.intp, redarrdim),
            "dtype": types.DType(reddtype),
            "order": types.literal(inputArrayType.layout),
            "device": types.literal(inputArrayType.device),
            "usm_type": types.literal(inputArrayType.usm_type),
        }
        glbl_np_empty = pfbdr.bind_global_function(
            fobj=fillFunc,
            ftype=get_np_ufunc_typ(fillFunc),
            args=[],
            kws=kws,
        )

        sizeVar = pfbdr.make_tuple_variable(
            [self.partial_sum_size_var], name="tuple_sizeVar"
        )
        cval = pfbdr._typingctx.resolve_value_type(reddtype)
        dt = pfbdr.make_const_variable(cval=cval, typ=types.DType(reddtype))

        orderTyVar = pfbdr.make_const_variable(
            cval=inputArrayType.layout, typ=types.literal(inputArrayType.layout)
        )

        deviceVar = pfbdr.make_const_variable(
            cval=inputArrayType.device,
            typ=types.literal(inputArrayType.device),
        )
        usmTyVar = pfbdr.make_const_variable(
            cval=inputArrayType.usm_type,
            typ=types.literal(inputArrayType.usm_type),
        )
        empty_call = pfbdr.call(
            glbl_np_empty, args=[sizeVar, dt, orderTyVar, deviceVar, usmTyVar]
        )

        self.partial_sum_var = pfbdr.assign(
            rhs=empty_call,
            typ=redarrvar_typ,
            name="partial_sum",
        )

        # final sum with size of 1
        final_sum_size = 1
        # writing work_group_size into IR
        final_sum_size_var = pfbdr.assign(
            rhs=ir.Const(final_sum_size, loc),
            typ=types.literal(final_sum_size),
            name="final_sum_size",
        )
        sizeVar = pfbdr.make_tuple_variable(
            [final_sum_size_var], name="tuple_sizeVar"
        )

        empty_call = pfbdr.call(
            glbl_np_empty, args=[sizeVar, dt, orderTyVar, deviceVar, usmTyVar]
        )
        self.final_sum_var = pfbdr.assign(
            rhs=empty_call,
            typ=redarrvar_typ,
            name="final_sum",
        )
        self.work_group_size = work_group_size
        self.redvars_to_redarrs_dict = {}
        self.redvars_to_redarrs_dict[red_name] = []
        self.redvars_to_redarrs_dict[red_name].append(self.partial_sum_var.name)
        self.redvars_to_redarrs_dict[red_name].append(self.final_sum_var.name)

    def _redtyp_to_redarraytype(self, redtyp, inputArrayType):
        """Go from a reduction variable type to a reduction array type
        used to hold per-worker results.
        """
        redarrdim = 1
        # If the reduction type is an array then allocate reduction array with
        # ndim+1 dimensions.
        if isinstance(redtyp, DpnpNdArray):
            redarrdim += redtyp.ndim
            # We don't create array of array but multi-dimensional reduction
            # array with same dtype.
            redtyp = redtyp.dtype
            red_layout = redtyp.layout
            red_usm_type = redtyp.usm_type
            red_device = redtyp.device
        else:
            redtyp = inputArrayType.dtype
            red_layout = inputArrayType.layout
            red_usm_type = inputArrayType.usm_type
            red_device = inputArrayType.device
        # For scalar, figure out from RHS what the array type is going to be
        # for partial sums
        # return type is the type for partial sum

        return DpnpNdArray(
            ndim=redarrdim,
            dtype=redtyp,
            layout=red_layout,
            usm_type=red_usm_type,
            device=red_device,
        )


class ReductionKernelVariables:
    """
    The parfor body and the main function body share ir.Var nodes.
    We have to do some replacements of Var names in the parfor body
    to make them legal parameter names. If we don't copy then the
    Vars in the main function also would incorrectly change their name.
    """

    def __init__(
        self,
        lowerer,
        parfor_node,
        typemap,
        parfor_outputs,
        reductionHelperList,
    ):
        races = parfor_node.races
        loop_body = copy.copy(parfor_node.loop_body)
        remove_dels(loop_body)

        # parfor_dim = len(parfor_node.loop_nests) # noqa: E800 help understanding
        loop_indices = [
            loop_nest.index_variable.name
            for loop_nest in parfor_node.loop_nests
        ]

        # Get all the parfor params.
        parfor_params = parfor_node.params

        # Get all parfor reduction vars, and operators.
        typemap = lowerer.fndesc.typemap

        parfor_redvars, parfor_reddict = parfor.get_parfor_reductions(
            lowerer.func_ir,
            parfor_node,
            parfor_params,
            lowerer.fndesc.calltypes,
        )
        # Compute just the parfor inputs as a set difference.
        parfor_inputs = sorted(list(set(parfor_params) - set(parfor_outputs)))
        from .kernel_builder import _replace_var_with_array

        _replace_var_with_array(
            races, loop_body, typemap, lowerer.fndesc.calltypes
        )

        # Reorder all the params so that inputs go first then outputs.
        parfor_params = parfor_inputs + parfor_outputs

        # Some Var and loop_indices may not have legal parameter names so create
        # a dict of potentially illegal param name to guaranteed legal name.
        param_dict = {}
        ind_dict = {}

        from .kernel_builder import _legalize_names_with_typemap

        param_dict = _legalize_names_with_typemap(parfor_params, typemap)
        ind_dict = _legalize_names_with_typemap(loop_indices, typemap)
        self._parfor_reddict = parfor_reddict
        # output of reduction
        self._parfor_redvars = parfor_redvars
        self._redvars_legal_dict = legalize_names(parfor_redvars)
        # Compute a new list of legal loop index names.
        legal_loop_indices = [ind_dict[v] for v in loop_indices]

        tmp1 = []
        # output of reduction computed on device
        self._final_sum_names = []
        self._parfor_redvars_to_redarrs = {}
        for ele1 in reductionHelperList:
            for key in ele1.redvars_to_redarrs_dict.keys():
                self._parfor_redvars_to_redarrs[key] = (
                    ele1.redvars_to_redarrs_dict[key]
                )
                tmp1.append(ele1.redvars_to_redarrs_dict[key][0])
                tmp1.append(ele1.redvars_to_redarrs_dict[key][1])
            self._final_sum_names.append(ele1.final_sum_var.name)

        self._parfor_redarrs_legal_dict = {}
        self._parfor_redarrs_legal_dict = legalize_names(tmp1)

        # Get the types of each parameter.
        from .kernel_builder import _to_scalar_from_0d

        param_types = [_to_scalar_from_0d(typemap[v]) for v in parfor_params]

        # Calculate types of args passed to the kernel function.
        func_arg_types = [typemap[v] for v in (parfor_inputs + parfor_outputs)]

        # Replace illegal parameter names in the loop body with legal ones.
        replace_var_names(loop_body, param_dict)

        # remember the name before legalizing as the actual arguments
        self.parfor_params = parfor_params
        # Change parfor_params to be legal names.
        parfor_params = [param_dict[v] for v in parfor_params]

        parfor_params_orig = parfor_params

        parfor_params = []
        ascontig = False
        for pindex in range(len(parfor_params_orig)):
            if (
                ascontig
                and pindex < len(parfor_inputs)
                and isinstance(param_types[pindex], types.npytypes.Array)
            ):
                parfor_params.append(parfor_params_orig[pindex] + "param")
            else:
                parfor_params.append(parfor_params_orig[pindex])

        # Change parfor body to replace illegal loop index vars with legal ones.
        replace_var_names(loop_body, ind_dict)
        self._legal_loop_indices = legal_loop_indices
        self._loop_body = loop_body
        self._func_arg_types = func_arg_types
        self._ind_dict = ind_dict
        self._param_dict = param_dict
        self._parfor_legalized_params = parfor_params
        self._param_types = param_types
        self._lowerer = lowerer
        self._work_group_size = reductionHelperList[0].work_group_size

    @property
    def parfor_reddict(self):
        return self._parfor_reddict

    @property
    def parfor_redvars(self):
        return self._parfor_redvars

    @property
    def redvars_legal_dict(self):
        return self._redvars_legal_dict

    @property
    def final_sum_names(self):
        return self._final_sum_names

    @property
    def parfor_redarrs_legal_dict(self):
        return self._parfor_redarrs_legal_dict

    @property
    def parfor_redvars_to_redarrs(self):
        return self._parfor_redvars_to_redarrs

    @property
    def legal_loop_indices(self):
        return self._legal_loop_indices

    @property
    def loop_body(self):
        return copy.deepcopy(self._loop_body)

    @property
    def func_arg_types(self):
        return self._func_arg_types

    @property
    def ind_dict(self):
        return self._ind_dict

    @property
    def param_dict(self):
        return self._param_dict

    @property
    def parfor_legalized_params(self):
        return self._parfor_legalized_params

    @property
    def param_types(self):
        return self._param_types

    @property
    def lowerer(self):
        return self._lowerer

    @property
    def work_group_size(self):
        return self._work_group_size

    def copy_final_sum_to_host(self, queue_ref):
        lowerer = self.lowerer
        builder = lowerer.builder
        context = lowerer.context

        for i, redvar in enumerate(self.parfor_redvars):
            srcVar = self.final_sum_names[i]

            item_size = builder.gep(
                lowerer.getvar(srcVar),
                [
                    context.get_constant(types.int32, 0),
                    context.get_constant(types.int32, 3),  # itmesize
                ],
            )

            array_attr = builder.gep(
                lowerer.getvar(srcVar),
                [
                    context.get_constant(types.int32, 0),
                    context.get_constant(types.int32, 4),  # data
                ],
            )

            dest = builder.bitcast(
                lowerer.getvar(redvar),
                get_llvm_type(context=context, type=types.voidptr),
            )
            src = builder.bitcast(
                builder.load(array_attr),
                get_llvm_type(context=context, type=types.voidptr),
            )

            args = [
                queue_ref,
                dest,
                src,
                builder.load(item_size),
            ]

            event_ref = sycl.dpctl_queue_memcpy(builder, *args)
            sycl.dpctl_event_wait(builder, event_ref)
            sycl.dpctl_event_delete(builder, event_ref)
