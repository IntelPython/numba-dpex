# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy

from numba.core import funcdesc, ir, types
from numba.core.compiler_machinery import LoweringPass, register_pass
from numba.core.lowering import Lower
from numba.parfors.parfor_lowering import (
    _lower_parfor_parallel as _lower_parfor_parallel_std,
)

from numba_dpex import config
from numba_dpex.dpctl_iface.kernel_launcher import KernelLauncher

from ..exceptions import UnsupportedParforError
from ..types.dpnp_ndarray_type import DpnpNdArray
from ..utils.kernel_builder import GufuncKernel, create_kernel_for_parfor
from .parfor import Parfor, find_potential_aliases_parfor, get_parfor_outputs

# A global list of kernels to keep the objects alive indefinitely.
keep_alive_kernels = []

config.DEBUG_ARRAY_OPT = True


def _getvar_or_none(lowerer, x):
    try:
        return lowerer.getvar(x)
    except:
        return None


def _loadvar_or_none(lowerer, x):
    try:
        return lowerer.loadvar(x)
    except:
        return None


def _val_type_or_none(context, lowerer, x):
    try:
        return context.get_value_type(lowerer.fndesc.typemap[x])
    except:
        return None


def _create_shape_signature(
    get_shape_classes,
    num_inputs,
    # num_reductions,
    args,
    func_sig,
    races,
    typemap,
):
    """Create shape signature for GUFunc"""
    if config.DEBUG_ARRAY_OPT:
        print("_create_shape_signature", num_inputs, args)
        arg_start_print = 0
        for i in args[arg_start_print:]:
            print("argument", i, type(i), get_shape_classes(i, typemap=typemap))

    # num_inouts = len(args) - num_reductions
    # num_inouts = len(args)
    # maximum class number for array shapes
    classes = [
        get_shape_classes(var, typemap=typemap) if var not in races else (-1,)
        for var in args[1:]
    ]
    class_set = set()
    for _class in classes:
        if _class:
            for i in _class:
                class_set.add(i)
    max_class = max(class_set) + 1 if class_set else 0
    classes.insert(0, (max_class,))  # force set the class of 'sched' argument
    class_set.add(max_class)
    class_map = {}
    # TODO: use prefix + class number instead of single char
    alphabet = ord("a")
    for n in class_set:
        if n >= 0:
            class_map[n] = chr(alphabet)
            alphabet += 1

    alpha_dict = {"latest_alpha": alphabet}

    def bump_alpha(c, class_map):
        if c >= 0:
            return class_map[c]
        else:
            alpha_dict["latest_alpha"] += 1
            return chr(alpha_dict["latest_alpha"])

    gu_sin = []
    gu_sout = []
    count = 0
    syms_sin = ()

    if config.DEBUG_ARRAY_OPT:
        print("args", args)
        print("classes", classes)

    for cls, arg in zip(classes, args):
        count = count + 1
        if cls:
            dim_syms = tuple(bump_alpha(c, class_map) for c in cls)
        else:
            dim_syms = ()
        gu_sin.append(dim_syms)
        syms_sin += dim_syms
    return (gu_sin, gu_sout)


def _submit_gufunc_kernel(
    lowerer,
    gufunc_kernel,
    gu_signature,
    num_inputs,
    loop_ranges,
):
    """
    Adds the call to the gufunc function from the main function.
    """

    context = lowerer.context
    sin, sout = gu_signature
    num_dim = len(loop_ranges)

    keep_alive_kernels.append(gufunc_kernel.kernel)

    # Helper class that generates the dpctl function calls
    kernel_launcher = KernelLauncher(lowerer, gufunc_kernel.kernel, num_inputs)

    # Create a local variable storing a pointer to a sycl queue
    curr_queue = kernel_launcher.get_queue(exec_queue=gufunc_kernel.queue)

    num_flattened_args = 0

    # Compute number of args to be passed to the kernel. Note that the
    # actual number of kernel arguments is greater than the count of
    # gufunc_kernel.kernel_args as arrays get flattened.
    for arg_type in gufunc_kernel.kernel_arg_types:
        if isinstance(arg_type, DpnpNdArray):
            # FIXME: Remove magic constants
            num_flattened_args += 5 + (2 * arg_type.ndim)
        else:
            num_flattened_args += 1

    # Create LLVM values for the kernel args list and kernel arg types list
    args_list, args_ty_list = kernel_launcher.allocate_kernel_arg_array(
        num_flattened_args
    )

    # Populate the args_list and the args_ty_list LLVM arrays
    kernel_arg_num = 0
    for arg_num, arg in enumerate(gufunc_kernel.kernel_args):
        argtype = gufunc_kernel.kernel_arg_types[arg_num]
        if isinstance(argtype, DpnpNdArray):
            kernel_launcher.build_array_arg(
                array_val=arg,
                array_rank=argtype.ndim,
                arg_list=args_list,
                args_ty_list=args_ty_list,
                arg_num=kernel_arg_num,
            )
            # FIXME: Get rid of magic constants
            kernel_arg_num += 5 + (2 * argtype.ndim)
        else:
            llvm_val = _getvar_or_none(lowerer, arg)
            if not llvm_val:
                raise AssertionError
            kernel_launcher.build_arg(
                llvm_val, argtype, args_list, args_ty_list, kernel_arg_num
            )
            kernel_arg_num += 1

    # ninouts = len(expr_args)

    # all_llvm_args = [_getvar_or_none(lowerer, x) for x in expr_args[:ninouts]]
    # all_val_types = [
    #     _val_type_or_none(context, lowerer, x) for x in expr_args[:ninouts]
    # ]
    # all_args = [_loadvar_or_none(lowerer, x) for x in expr_args[:ninouts]]

    # # Call clSetKernelArg for each arg and create arg array for
    # # the enqueue function. Put each part of each argument into
    # # kernel_arg_array.
    # for var, llvm_arg, arg_type, gu_sig, val_type, index in zip(
    #     expr_args,
    #     all_llvm_args,
    #     expr_arg_types,
    #     sin + sout,
    #     all_val_types,
    #     range(len(expr_args)),
    # ):
    #     kernel_launcher.process_kernel_arg(
    #         var, llvm_arg, arg_type, index, modified_arrays, curr_queue
    #     )

    # loadvars for loop_ranges
    def load_range(v):
        if isinstance(v, ir.Var):
            return lowerer.loadvar(v.name)
        else:
            return context.get_constant(types.uintp, v)

    num_dim = len(loop_ranges)
    for i in range(num_dim):
        start, stop, step = loop_ranges[i]
        start = load_range(start)
        stop = load_range(stop)
        assert step == 1  # We do not support loop steps other than 1
        step = load_range(step)
        loop_ranges[i] = (start, stop, step)

    kernel_launcher.submit_sync_ranged_kernel(loop_ranges, curr_queue)

    # At this point we can free the DPCTLSyclQueueRef (curr_queue)
    kernel_launcher.free_queue(sycl_queue_val=curr_queue)


def _lower_parfor_gufunc(lowerer, parfor):
    """Lowers a parfor node created by the dpjit compiler to a kernel.

    The general approach is as follows:

        - The code from the parfor's init block is lowered normally
        in the context of the current function.
        - The body of the parfor is transformed into a gufunc function.

    """
    # We copy the typemap here because for race condition variable we'll
    # update their type to array so they can be updated by the gufunc.
    orig_typemap = lowerer.fndesc.typemap

    # replace original typemap with copy and restore the original at the end.
    lowerer.fndesc.typemap = copy.copy(orig_typemap)

    if config.DEBUG_ARRAY_OPT:
        print("lowerer.fndesc", lowerer.fndesc, type(lowerer.fndesc))

    typemap = lowerer.fndesc.typemap
    varmap = lowerer.varmap

    loc = parfor.init_block.loc
    scope = parfor.init_block.scope

    # Lower the init block of the parfor.
    for instr in parfor.init_block.body:
        lowerer.lower_inst(instr)
        print(instr)
        print(varmap)

    for racevar in parfor.races:
        if racevar not in varmap:
            rvtyp = typemap[racevar]
            rv = ir.Var(scope, racevar, loc)
            lowerer._alloca_var(rv.name, rvtyp)

    alias_map = {}
    arg_aliases = {}

    find_potential_aliases_parfor(
        parfor, parfor.params, typemap, lowerer.func_ir, alias_map, arg_aliases
    )

    # run get_parfor_outputs() and get_parfor_reductions() before
    # gufunc creation since Jumps are modified so CFG of loop_body
    # dict will become invalid
    if parfor.params is None:
        raise AssertionError

    parfor_output_arrays = get_parfor_outputs(parfor, parfor.params)

    # compile parfor body as a separate function to be used with GUFuncWrapper
    flags = copy.copy(parfor.flags)
    flags.error_model = "numpy"

    # Can't get here unless flags.set('auto_parallel', ParallelOptions(True))
    index_var_typ = typemap[parfor.loop_nests[0].index_variable.name]

    # index variables should have the same type, check rest of indices
    for loop_nest in parfor.loop_nests[1:]:
        if typemap[loop_nest.index_variable.name] != index_var_typ:
            raise AssertionError

    loop_ranges = [
        (loop_nest.start, loop_nest.stop, loop_nest.step)
        for loop_nest in parfor.loop_nests
    ]

    try:
        gufunc_kernel = create_kernel_for_parfor(
            lowerer,
            parfor,
            typemap,
            flags,
            loop_ranges,
            bool(alias_map),
            parfor.races,
            parfor_output_arrays,
        )
    except Exception:
        # FIXME: Make the exception more informative
        raise UnsupportedParforError

    num_inputs = len(gufunc_kernel.kernel_args) - len(parfor_output_arrays)
    gu_signature = _create_shape_signature(
        parfor.get_shape_classes,
        num_inputs,
        gufunc_kernel.kernel_args,
        gufunc_kernel.signature,
        parfor.races,
        typemap,
    )

    _submit_gufunc_kernel(
        lowerer,
        gufunc_kernel,
        gu_signature,
        num_inputs,
        loop_ranges,
    )

    # Restore the original typemap of the function that was replaced
    # temporarily at the beginning of this function.
    lowerer.fndesc.typemap = orig_typemap


class WrapperDefaultLower(Lower):
    @property
    def _disable_sroa_like_opt(self):
        """We always return True."""
        return True


class _ParforLower(Lower):
    """Extends standard lowering to accommodate parfor.Parfor nodes that may
    have the `lowerer` attribute set.
    """

    def __init__(self, context, library, fndesc, func_ir, metadata=None):
        Lower.__init__(self, context, library, fndesc, func_ir, metadata)
        self.dpex_lower = self._lower(
            context, library, fndesc, func_ir, metadata
        )

    def _lower(self, context, library, fndesc, func_ir, metadata):
        """Create Lower with changed linkageName in debug info"""
        lower = WrapperDefaultLower(context, library, fndesc, func_ir, metadata)

        # Debuginfo
        if context.enable_debuginfo:
            from numba.core.funcdesc import default_mangler, qualifying_prefix

            from numba_dpex.debuginfo import DpexDIBuilder

            qualprefix = qualifying_prefix(fndesc.modname, fndesc.qualname)
            mangled_qualname = default_mangler(qualprefix, fndesc.argtypes)

            lower.debuginfo = DpexDIBuilder(
                module=lower.module,
                filepath=func_ir.loc.filename,
                linkage_name=mangled_qualname,
                cgctx=context,
            )

        return lower

    def lower(self):
        context = self.dpex_lower.context

        # Only Numba's CPUContext has the `lower_extension` attribute
        context.lower_extensions[Parfor] = lower_parfor_dpex
        self.dpex_lower.lower()
        self.base_lower = self.dpex_lower

        self.env = self.base_lower.env
        self.call_helper = self.base_lower.call_helper

    def create_cpython_wrapper(self, release_gil=False):
        return self.base_lower.create_cpython_wrapper(release_gil)


def lower_parfor_dpex(lowerer, parfor):
    # FIXME: Temporary for testing
    parfor.lowerer = _lower_parfor_gufunc

    if parfor.lowerer is None:
        _lower_parfor_parallel_std(lowerer, parfor)
    else:
        parfor.lowerer(lowerer, parfor)


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforLoweringPass(LoweringPass):
    """A custom lowering pass that does dpex-specific lowering of parfor
    nodes.

    FIXME: Redesign once numba-dpex supports Numba 0.57
    """

    _name = "dpjit_lowering"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        targetctx = state.targetctx

        library = state.library
        interp = state.func_ir
        typemap = state.typemap
        restype = state.return_type
        calltypes = state.calltypes
        flags = state.flags
        metadata = state.metadata

        kwargs = {}

        # for support numba 0.54 and <=0.55.0dev0=*_469
        if hasattr(flags, "get_mangle_string"):
            kwargs["abi_tags"] = flags.get_mangle_string()
        # Lowering
        fndesc = funcdesc.PythonFunctionDescriptor.from_specialized_function(
            interp,
            typemap,
            restype,
            calltypes,
            mangler=targetctx.mangler,
            inline=flags.forceinline,
            noalias=flags.noalias,
            **kwargs,
        )

        with targetctx.push_code_library(library):
            lower = _ParforLower(
                targetctx, library, fndesc, interp, metadata=metadata
            )
            lower.lower()
            if not flags.no_cpython_wrapper:
                lower.create_cpython_wrapper(flags.release_gil)

            env = lower.env
            call_helper = lower.call_helper
            del lower

        from numba.core.compiler import _LowerResult  # TODO: move this

        if flags.no_compile:
            state["cr"] = _LowerResult(fndesc, call_helper, cfunc=None, env=env)
        else:
            # Prepare for execution
            cfunc = targetctx.get_executable(library, fndesc, env)
            # Insert native function for use by other jitted-functions.
            # We also register its library to allow for inlining.
            targetctx.insert_user_function(cfunc, fndesc, [library])
            state["cr"] = _LowerResult(
                fndesc, call_helper, cfunc=cfunc, env=env
            )

        return True
