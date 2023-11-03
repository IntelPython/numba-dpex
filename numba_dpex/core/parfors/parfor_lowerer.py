# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import namedtuple

from llvmlite import ir as llvmir
from numba.core import cgutils, ir, types
from numba.parfors.parfor import (
    find_potential_aliases_parfor,
    get_parfor_outputs,
)

from numba_dpex import config
from numba_dpex.core.datamodel.models import dpex_data_model_manager as dpex_dmm
from numba_dpex.core.parfors.reduction_helper import (
    ReductionHelper,
    ReductionKernelVariables,
)
from numba_dpex.core.utils.kernel_launcher import KernelLaunchIRBuilder

from ..exceptions import UnsupportedParforError
from ..types.dpnp_ndarray_type import DpnpNdArray
from .kernel_builder import create_kernel_for_parfor
from .reduction_kernel_builder import (
    create_reduction_main_kernel_for_parfor,
    create_reduction_remainder_kernel_for_parfor,
)

_KernelArgs = namedtuple(
    "_KernelArgs",
    ["num_flattened_args", "arg_vals", "arg_types"],
)


# A global list of kernels to keep the objects alive indefinitely.
keep_alive_kernels = []


def _getvar(lowerer, x):
    """Returns the LLVM Value corresponding to a Numba IR variable.

    Depending on if Numba's sroa-like optimization is enabled or not, the
    LLVM value for an Numba IR variable is found in either the ``varmap``
    or the ``blk_local_varmap`` of the ``lowerer``. If the LLVM Value is not a
    pointer, e.g., in case of function args with sroa optimization enabled, then
    creates an alloca and stores the Value into the new alloca Value and returns
    it. The extra alloca is needed as all inputs to a kernel function need to
    be passed by reference and not value.

    Args:
        lowerer: The Numba Lower instance used to lower the function.
        x: Numba IR variable name used to lookup the corresponding
           LLVM Value.

    Raises:
        AssertionError: If the LLVM Value for ``x`` does not exist in either
            the ``varmap`` or the ``blk_local_varmap``.

    Returns: An LLVM Value object

    """
    var_val = None
    if x in lowerer._blk_local_varmap:
        var_val = lowerer._blk_local_varmap[x]
    elif x in lowerer.varmap:
        var_val = lowerer.varmap[x]

    if var_val:
        if not isinstance(var_val.type, llvmir.PointerType):
            with lowerer.builder.goto_entry_block():
                var_val_ptr = lowerer.builder.alloca(var_val.type)
            lowerer.builder.store(var_val, var_val_ptr)
            return var_val_ptr
        else:
            return var_val
    else:
        raise AssertionError("No llvm Value found for kernel arg")


def _load_range(lowerer, value):
    if isinstance(value, ir.Var):
        return lowerer.loadvar(value.name)
    else:
        return lowerer.context.get_constant(types.uintp, value)


class ParforLowerImpl:
    """Provides a custom lowerer for parfor nodes that generates a SYCL kernel
    for a parfor and submits it to a queue.
    """

    def _build_kernel_arglist(self, kernel_fn, lowerer, kernel_builder):
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
                datamodel = dpex_dmm.lookup(arg_type)
                num_flattened_args += datamodel.flattened_field_count
            elif arg_type == types.complex64 or arg_type == types.complex128:
                num_flattened_args += 2
            else:
                num_flattened_args += 1

        # Create LLVM values for the kernel args list and kernel arg types list
        args_list = kernel_builder.allocate_kernel_arg_array(num_flattened_args)
        args_ty_list = kernel_builder.allocate_kernel_arg_ty_array(
            num_flattened_args
        )
        callargs_ptrs = []
        for arg in kernel_fn.kernel_args:
            callargs_ptrs.append(_getvar(lowerer, arg))

        kernel_builder.populate_kernel_args_and_args_ty_arrays(
            kernel_argtys=kernel_fn.kernel_arg_types,
            callargs_ptrs=callargs_ptrs,
            args_list=args_list,
            args_ty_list=args_ty_list,
            datamodel_mgr=dpex_dmm,
        )

        return _KernelArgs(
            num_flattened_args=num_flattened_args,
            arg_vals=args_list,
            arg_types=args_ty_list,
        )

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
        kernel_builder = KernelLaunchIRBuilder(lowerer.context, lowerer.builder)

        ptr_to_queue_ref = kernel_builder.get_queue(exec_queue=kernel_fn.queue)
        args = self._build_kernel_arglist(kernel_fn, lowerer, kernel_builder)

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

        kernel_ref_addr = kernel_fn.kernel.addressof_ref()
        kernel_ref = lowerer.builder.inttoptr(
            lowerer.context.get_constant(types.uintp, kernel_ref_addr),
            cgutils.voidptr_t,
        )
        curr_queue_ref = lowerer.builder.load(ptr_to_queue_ref)

        # Submit a synchronous kernel
        kernel_builder.submit_sycl_kernel(
            sycl_kernel_ref=kernel_ref,
            sycl_queue_ref=curr_queue_ref,
            total_kernel_args=args.num_flattened_args,
            arg_list=args.arg_vals,
            arg_ty_list=args.arg_types,
            global_range=global_range,
            local_range=local_range,
        )

        # At this point we can free the DPCTLSyclQueueRef (curr_queue)
        kernel_builder.free_queue(ptr_to_sycl_queue_ref=ptr_to_queue_ref)

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
        kernel_builder = KernelLaunchIRBuilder(lowerer.context, lowerer.builder)

        ptr_to_queue_ref = kernel_builder.get_queue(exec_queue=kernel_fn.queue)

        args = self._build_kernel_arglist(kernel_fn, lowerer, kernel_builder)
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

        kernel_ref_addr = kernel_fn.kernel.addressof_ref()
        kernel_ref = lowerer.builder.inttoptr(
            lowerer.context.get_constant(types.uintp, kernel_ref_addr),
            cgutils.voidptr_t,
        )
        curr_queue_ref = lowerer.builder.load(ptr_to_queue_ref)

        # Submit a synchronous kernel
        kernel_builder.submit_sycl_kernel(
            sycl_kernel_ref=kernel_ref,
            sycl_queue_ref=curr_queue_ref,
            total_kernel_args=args.num_flattened_args,
            arg_list=args.arg_vals,
            arg_ty_list=args.arg_types,
            global_range=global_range,
            local_range=local_range,
        )

        # At this point we can free the DPCTLSyclQueueRef (curr_queue)
        kernel_builder.free_queue(ptr_to_sycl_queue_ref=ptr_to_queue_ref)

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

        kernel_builder = KernelLaunchIRBuilder(lowerer.context, lowerer.builder)

        ptr_to_queue_ref = kernel_builder.get_queue(exec_queue=kernel_fn.queue)

        args = self._build_kernel_arglist(kernel_fn, lowerer, kernel_builder)
        # Create a global range over which to submit the kernel based on the
        # loop_ranges of the parfor
        global_range = []

        stop = _load_range(lowerer, 1)

        global_range.append(stop)

        local_range = []

        kernel_ref_addr = kernel_fn.kernel.addressof_ref()
        kernel_ref = lowerer.builder.inttoptr(
            lowerer.context.get_constant(types.uintp, kernel_ref_addr),
            cgutils.voidptr_t,
        )
        curr_queue_ref = lowerer.builder.load(ptr_to_queue_ref)

        # Submit a synchronous kernel
        kernel_builder.submit_sycl_kernel(
            sycl_kernel_ref=kernel_ref,
            sycl_queue_ref=curr_queue_ref,
            total_kernel_args=args.num_flattened_args,
            arg_list=args.arg_vals,
            arg_ty_list=args.arg_types,
            global_range=global_range,
            local_range=local_range,
        )

        # At this point we can free the DPCTLSyclQueueRef (curr_queue)
        kernel_builder.free_queue(ptr_to_sycl_queue_ref=ptr_to_queue_ref)

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
        # flags.set('auto_parallel', ParallelOptions(True)) # noqa: E800
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

    Each Parfor instruction can have an optional "lowerer" attribute. The
    lowerer attribute determines how the parfor instruction should be lowered
    to LLVM IR. In addition, the lower attribute decides which parfor
    instructions can be fused together.

    The factory class maintains a dictionary mapping every device
    type (filter string) encountered so far to a lowerer function for that
    device type. At this point numba-dpex does not generate device-specific code
    and the lowerer used is same for all device types. However, as a different
    ParforLowerImpl instance is returned for every parfor instruction that has
    a distinct compute-follows-data inferred device it prevents illegal
    parfor fusion.
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
