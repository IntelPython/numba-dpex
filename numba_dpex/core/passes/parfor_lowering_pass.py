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
from numba_dpex.core.utils.kernel_launcher import KernelLaunchIRBuilder
from numba_dpex.core.utils.reduction_helper import (
    ReductionHelper,
    ReductionKernelVariables,
)

from ..exceptions import UnsupportedParforError
from ..types.dpnp_ndarray_type import DpnpNdArray
from ..utils.kernel_builder import create_kernel_for_parfor
from ..utils.reduction_kernel_builder import (
    create_reduction_main_kernel_for_parfor,
    create_reduction_remainder_kernel_for_parfor,
)
from .parfor import Parfor, find_potential_aliases_parfor, get_parfor_outputs

# A global list of kernels to keep the objects alive indefinitely.
keep_alive_kernels = []


def _getvar_or_none(lowerer, x):
    try:
        return lowerer.getvar(x)
    except:
        return None


def _load_range(lowerer, value):
    if isinstance(value, ir.Var):
        return lowerer.loadvar(value.name)
    else:
        return lowerer.context.get_constant(types.uintp, value)


class ParforLowerImpl:
    """Provides a custom lowerer for parfor nodes that generates a SYCL kernel
    for a parfor and submits it to a queue.
    """

    def _get_exec_queue(self, kernel_fn, lowerer):
        """Creates a stack variable storing the sycl queue pointer used to
        launch the kernel function.
        """
        self.kernel_builder = KernelLaunchIRBuilder(lowerer, kernel_fn.kernel)

        # Create a local variable storing a pointer to a DPCTLSyclQueueRef
        # pointer.
        self.curr_queue = self.kernel_builder.get_queue(
            exec_queue=kernel_fn.queue
        )

    def _build_kernel_arglist(self, kernel_fn, lowerer):
        """Creates local variables for all the arguments and the argument types
        that are passes to the kernel function.

        Args:
            kernel_fn: Kernel function to be launched.
            lowerer: The Numba lowerer used to generate the LLVM IR

        Raises:
            AssertionError: If the LLVM IR Value for an argument defined in
            Numba IR is not found.
        """
        num_flattened_args = 0

        # Compute number of args to be passed to the kernel. Note that the
        # actual number of kernel arguments is greater than the count of
        # kernel_fn.kernel_args as arrays get flattened.
        for arg_type in kernel_fn.kernel_arg_types:
            if isinstance(arg_type, DpnpNdArray):
                # FIXME: Remove magic constants
                num_flattened_args += 5 + (2 * arg_type.ndim)
            elif arg_type == types.complex64 or arg_type == types.complex128:
                num_flattened_args += 2
            else:
                num_flattened_args += 1

        # Create LLVM values for the kernel args list and kernel arg types list
        self.args_list = self.kernel_builder.allocate_kernel_arg_array(
            num_flattened_args
        )
        self.args_ty_list = self.kernel_builder.allocate_kernel_arg_ty_array(
            num_flattened_args
        )

        # Populate the args_list and the args_ty_list LLVM arrays
        self.kernel_arg_num = 0
        for arg_num, arg in enumerate(kernel_fn.kernel_args):
            argtype = kernel_fn.kernel_arg_types[arg_num]
            llvm_val = _getvar_or_none(lowerer, arg)
            if not llvm_val:
                raise AssertionError
            if isinstance(argtype, DpnpNdArray):
                self.kernel_builder.build_array_arg(
                    array_val=llvm_val,
                    array_rank=argtype.ndim,
                    arg_list=self.args_list,
                    args_ty_list=self.args_ty_list,
                    arg_num=self.kernel_arg_num,
                )
                # FIXME: Get rid of magic constants
                self.kernel_arg_num += 5 + (2 * argtype.ndim)
            else:
                if argtype == types.complex64:
                    self.kernel_builder.build_complex_arg(
                        llvm_val,
                        types.float32,
                        self.args_list,
                        self.args_ty_list,
                        self.kernel_arg_num,
                    )
                    self.kernel_arg_num += 2
                elif argtype == types.complex128:
                    self.kernel_builder.build_complex_arg(
                        llvm_val,
                        types.float64,
                        self.args_list,
                        self.args_ty_list,
                        self.kernel_arg_num,
                    )
                    self.kernel_arg_num += 2
                else:
                    self.kernel_builder.build_arg(
                        llvm_val,
                        argtype,
                        self.args_list,
                        self.args_ty_list,
                        self.kernel_arg_num,
                    )
                    self.kernel_arg_num += 1

    def _submit_parfor_kernel(
        self,
        lowerer,
        kernel_fn,
        loop_ranges,
    ):
        """
        Adds a call to submit a kernel function into the function body of the
        current Numba JIT compiled function.
        """
        # Ensure that the Python arguments are kept alive for the duration of
        # the kernel execution
        keep_alive_kernels.append(kernel_fn.kernel)

        self._get_exec_queue(kernel_fn, lowerer)
        self._build_kernel_arglist(kernel_fn, lowerer)
        # Create a global range over which to submit the kernel based on the
        # loop_ranges of the parfor
        global_range = []
        # SYCL ranges can have at max 3 dimension. If the parfor is of a higher
        # dimension then the indexing for the higher dimensions is done inside
        # the kernel.
        global_range_rank = len(loop_ranges) if len(loop_ranges) < 3 else 3

        for i in range(global_range_rank):
            start, stop, step = loop_ranges[i]
            stop = _load_range(lowerer, stop)
            if step != 1:
                raise UnsupportedParforError(
                    "non-unit strides are not yet supported."
                )
            global_range.append(stop)

        local_range = []

        # Submit a synchronous kernel
        self.kernel_builder.submit_sync_kernel(
            self.curr_queue,
            self.kernel_arg_num,
            self.args_list,
            self.args_ty_list,
            global_range,
            local_range,
        )

        # At this point we can free the DPCTLSyclQueueRef (curr_queue)
        self.kernel_builder.free_queue(sycl_queue_val=self.curr_queue)

    def _submit_reduction_main_parfor_kernel(
        self,
        lowerer,
        kernel_fn,
        reductionHelper=None,
    ):
        """
        Adds a call to submit the main kernel of a parfor reduction into the
        function body of the current Numba JIT compiled function.
        """
        # Ensure that the Python arguments are kept alive for the duration of
        # the kernel execution
        keep_alive_kernels.append(kernel_fn.kernel)

        self._get_exec_queue(kernel_fn, lowerer)
        self._build_kernel_arglist(kernel_fn, lowerer)
        # Create a global range over which to submit the kernel based on the
        # loop_ranges of the parfor
        global_range = []

        stop = reductionHelper.global_size_var
        stop = _load_range(lowerer, stop)
        global_range.append(stop)

        local_range = []
        local_range.append(
            _load_range(lowerer, reductionHelper.work_group_size)
        )

        # Submit a synchronous kernel
        self.kernel_builder.submit_sync_kernel(
            self.curr_queue,
            self.kernel_arg_num,
            self.args_list,
            self.args_ty_list,
            global_range,
            local_range,
        )

    def _submit_reduction_remainder_parfor_kernel(
        self,
        lowerer,
        kernel_fn,
    ):
        """
        Adds a call to submit the remainder kernel of a parfor reduction into
        the function body of the current Numba JIT compiled function.
        """
        # Ensure that the Python arguments are kept alive for the duration of
        # the kernel execution
        keep_alive_kernels.append(kernel_fn.kernel)

        self._get_exec_queue(kernel_fn, lowerer)
        self._build_kernel_arglist(kernel_fn, lowerer)
        # Create a global range over which to submit the kernel based on the
        # loop_ranges of the parfor
        global_range = []

        stop = _load_range(lowerer, 1)

        global_range.append(stop)

        local_range = []

        # Submit a synchronous kernel
        self.kernel_builder.submit_sync_kernel(
            self.curr_queue,
            self.kernel_arg_num,
            self.args_list,
            self.args_ty_list,
            global_range,
            local_range,
        )

    def _reduction_codegen(
        self,
        parfor,
        typemap,
        nredvars,
        parfor_redvars,
        parfor_reddict,
        lowerer,
        parfor_output_arrays,
        loop_ranges,
        flags,
        alias_map,
    ):
        """
        Reduction kernel generation and submission.
        """
        inputArrayName = None
        for para in parfor.params:
            if not isinstance(typemap[para], DpnpNdArray):
                continue
            inputArrayName = para
            break

        if inputArrayName is None:
            raise AssertionError

        inputArrayType = typemap[inputArrayName]

        reductionHelperList = []
        for i in range(nredvars):
            reductionHelper = ReductionHelper()
            reductionHelper._allocate_partial_reduction_arrays(
                parfor,
                lowerer,
                parfor_redvars[i],
                inputArrayType,
            )
            reductionHelperList.append(reductionHelper)

        reductionKernelVar = ReductionKernelVariables(
            lowerer=lowerer,
            parfor_node=parfor,
            typemap=typemap,
            parfor_outputs=parfor_output_arrays,
            reductionHelperList=reductionHelperList,
        )

        if len(parfor.loop_nests) > 1:
            raise UnsupportedParforError(
                "Reduction with nested loop is not yet supported."
            )

        parfor_kernel = create_reduction_main_kernel_for_parfor(
            loop_ranges,
            parfor,
            typemap,
            flags,
            bool(alias_map),
            reductionKernelVar,
            parfor_reddict,
        )

        self._submit_reduction_main_parfor_kernel(
            lowerer,
            parfor_kernel,
            reductionHelperList[0],
        )

        parfor_kernel = create_reduction_remainder_kernel_for_parfor(
            parfor,
            typemap,
            flags,
            bool(alias_map),
            reductionKernelVar,
            parfor_reddict,
            reductionHelperList,
        )

        self._submit_reduction_remainder_parfor_kernel(
            lowerer,
            parfor_kernel,
        )

        reductionKernelVar.copy_final_sum_to_host(parfor_kernel)

    def _lower_parfor_as_kernel(self, lowerer, parfor):
        """Lowers a parfor node created by the dpjit compiler to a
        ``numba_dpex.kernel``.

        The general approach is as follows:

            - The code from the parfor's init block is lowered normally
              in the context of the current function.
            - The body of the parfor is transformed into a kernel function.
            - Dpctl runtime calls to submit the kernel are added.

        """
        # We copy the typemap here because for race condition variable we'll
        # update their type to array so they can be updated by the kernel.
        orig_typemap = lowerer.fndesc.typemap

        # replace original typemap with copy and restore the original at the
        # end.
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

        for racevar in parfor.races:
            if racevar not in varmap:
                rvtyp = typemap[racevar]
                rv = ir.Var(scope, racevar, loc)
                lowerer._alloca_var(rv.name, rvtyp)

        alias_map = {}
        arg_aliases = {}

        find_potential_aliases_parfor(
            parfor,
            parfor.params,
            typemap,
            lowerer.func_ir,
            alias_map,
            arg_aliases,
        )

        # run get_parfor_outputs() and get_parfor_reductions() before
        # kernel creation since Jumps are modified so CFG of loop_body
        # dict will become invalid
        if parfor.params is None:
            raise AssertionError

        parfor_output_arrays = get_parfor_outputs(parfor, parfor.params)

        # compile parfor body as a separate dpex kernel function
        flags = copy.copy(parfor.flags)
        flags.error_model = "numpy"

        # Can't get here unless
        # flags.set('auto_parallel', ParallelOptions(True))
        index_var_typ = typemap[parfor.loop_nests[0].index_variable.name]

        # index variables should have the same type, check rest of indices
        for loop_nest in parfor.loop_nests[1:]:
            if typemap[loop_nest.index_variable.name] != index_var_typ:
                raise AssertionError

        loop_ranges = [
            (loop_nest.start, loop_nest.stop, loop_nest.step)
            for loop_nest in parfor.loop_nests
        ]

        parfor_redvars, parfor_reddict = parfor.redvars, parfor.reddict

        nredvars = len(parfor_redvars)
        if nredvars > 0:
            self._reduction_codegen(
                parfor,
                typemap,
                nredvars,
                parfor_redvars,
                parfor_reddict,
                lowerer,
                parfor_output_arrays,
                loop_ranges,
                flags,
                alias_map,
            )
        else:
            try:
                parfor_kernel = create_kernel_for_parfor(
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

            # Finally submit the kernel
            self._submit_parfor_kernel(
                lowerer,
                parfor_kernel,
                loop_ranges,
            )

        # TODO: free the kernel at this point

        # Restore the original typemap of the function that was replaced
        # temporarily at the beginning of this function.
        lowerer.fndesc.typemap = orig_typemap


class ParforLowerFactory:
    """A pseudo-factory class that maps a device filter string to a lowering
    function.

    Each parfor can have a "lowerer" attribute that determines how the parfor
    node is to be lowered to LLVM IR. The factory class maintains a static map
    that for every device type (filter string) encountered so far to a lowerer
    function for that device type. At this point numba-dpex does not generate
    device-specific code and there lowerer is always same for all devices.
    By generating different instances we make sure prfors that will execute on
    distinct devices as determined by compute-follows-data programming model are
    never fused together.
    """

    device_to_lowerer_map = {}

    @classmethod
    def get_lowerer(cls, device):
        try:
            lowerer = ParforLowerFactory.device_to_lowerer_map[device]
        except KeyError:
            lowerer = ParforLowerImpl()._lower_parfor_as_kernel
            ParforLowerFactory.device_to_lowerer_map[device] = lowerer

        return lowerer


class WrapperDefaultLower(Lower):
    @property
    def _disable_sroa_like_opt(self):
        """We always return True."""
        return True


def lower_parfor_dpex(lowerer, parfor):
    parfor.lowerer = ParforLowerImpl()._lower_parfor_as_kernel
    if parfor.lowerer is None:
        _lower_parfor_parallel_std(lowerer, parfor)
    else:
        parfor.lowerer(lowerer, parfor)


class _ParforLower(Lower):
    """Extends standard lowering to accommodate parfor.Parfor nodes that may
    have the ``lowerer`` attribute set.
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
