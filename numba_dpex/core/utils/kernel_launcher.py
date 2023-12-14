# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Module that contains numba style wrapper around sycl kernel submit."""

from dataclasses import dataclass

import dpctl
from llvmlite import ir as llvmir
from llvmlite.ir.builder import IRBuilder
from numba.core import cgutils, types
from numba.core.cpu import CPUContext
from numba.core.datamodel import DataModelManager

from numba_dpex import config, utils
from numba_dpex.core.runtime.context import DpexRTContext
from numba_dpex.core.types import DpnpNdArray
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl
from numba_dpex.dpctl_iface._helpers import numba_type_to_dpctl_typenum

MAX_SIZE_OF_SYCL_RANGE = 3


@dataclass
class _KernelLaunchIRArguments:  # pylint: disable=too-many-instance-attributes
    """List of kernel launch arguments used in sycl.dpctl_queue_submit_range and
    sycl.dpctl_queue_submit_ndrange."""

    sycl_kernel_ref: llvmir.Instruction = None
    sycl_queue_ref: llvmir.Instruction = None
    arg_list: llvmir.Instruction = None
    arg_ty_list: llvmir.Instruction = None
    total_kernel_args: llvmir.Instruction = None
    global_range: llvmir.Instruction = None
    local_range: llvmir.Instruction = None
    range_size: llvmir.Instruction = None
    dep_events: llvmir.Instruction = None
    dep_events_len: llvmir.Instruction = None

    def to_list(self):
        """Returns list of arguments in the right order to pass to
        sycl.dpctl_queue_submit_range or sycl.dpctl_queue_submit_ndrange."""
        res = [
            self.sycl_kernel_ref,
            self.sycl_queue_ref,
            self.arg_list,
            self.arg_ty_list,
            self.total_kernel_args,
            self.global_range,
        ]

        if self.local_range is not None:
            res += [self.local_range]

        res += [
            self.range_size,
            self.dep_events,
            self.dep_events_len,
        ]

        return res


class KernelLaunchIRBuilder:
    """
    KernelLaunchIRBuilder(lowerer, cres)
    Helper class to build the LLVM IR for the submission of a kernel.

    The class generates LLVM IR inside the current LLVM module that is needed
    for submitting kernels. The LLVM Values that
    """

    def __init__(
        self,
        context: CPUContext,
        builder: IRBuilder,
        kernel_dmm: DataModelManager,
    ):
        """Create a KernelLauncher for the specified kernel.

        Args:
            context: A Numba target context that will be used to generate the
                     code.
            builder: An llvmlite IRBuilder instance used to generate LLVM IR.
        """
        self.context = context
        self.builder = builder
        self.rtctx = DpexRTContext(self.context)
        self.arguments = _KernelLaunchIRArguments()
        self.kernel_dmm = kernel_dmm

    def _build_nullptr(self):
        """Builds the LLVM IR to represent a null pointer.

        Returns: An LLVM Value storing a null pointer
        """
        zero = cgutils.alloca_once(self.builder, utils.LLVMTypes.int64_t)
        self.builder.store(self.context.get_constant(types.int64, 0), zero)
        return self.builder.bitcast(
            zero, utils.get_llvm_type(context=self.context, type=types.voidptr)
        )

    def _build_array_attr_arg(  # pylint: disable=too-many-arguments
        self,
        array_val,
        array_attr_pos,
        array_attr_ty,
        arg_list,
        args_ty_list,
        arg_num,
    ):
        array_attr = self.builder.gep(
            array_val,
            [
                self.context.get_constant(types.int32, 0),
                self.context.get_constant(types.int32, array_attr_pos),
            ],
        )

        # FIXME: If pointer arg then load it to some value and pass that value.
        # We also most likely need an address space cast
        if isinstance(
            array_attr_ty, (types.misc.RawPointer, types.misc.CPointer)
        ):
            array_attr = self.builder.load(array_attr)

        self._build_arg(
            val=array_attr,
            ty=array_attr_ty,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )

    def _build_unituple_member_arg(  # pylint: disable=too-many-arguments
        self, array_val, array_attr_pos, ndims, arg_list, args_ty_list, arg_num
    ):
        array_attr = self.builder.gep(
            array_val,
            [
                self.context.get_constant(types.int32, 0),
                self.context.get_constant(types.int32, array_attr_pos),
            ],
        )

        for ndim in range(ndims):
            self._build_array_attr_arg(
                array_val=array_attr,
                array_attr_pos=ndim,
                array_attr_ty=types.int64,
                arg_list=arg_list,
                args_ty_list=args_ty_list,
                arg_num=arg_num + ndim,
            )

    def _build_arg(
        self, val, ty, arg_list, args_ty_list, arg_num
    ):  # pylint: disable=too-many-arguments
        """Stores the kernel arguments and the kernel argument types into
        arrays that will be passed to DPCTLQueue_SubmitRange.

        Args:
            val: An LLVM IR Value that will be stored into the arguments array
            ty: A Numba type that will be converted to a DPCTLKernelArgType
            enum and stored into the argument types list array
            arg_list: An LLVM IR Value array that stores the kernel arguments
            args_ty_list: An LLVM IR Value array that stores the
            DPCTLKernelArgType enum value for each kernel argument
            arg_num: The index position at which the arg_list and args_ty_list
            need to be updated.
        """
        kernel_arg_dst = self.builder.gep(
            arg_list,
            [self.context.get_constant(types.int32, arg_num)],
        )
        kernel_arg_ty_dst = self.builder.gep(
            args_ty_list,
            [self.context.get_constant(types.int32, arg_num)],
        )
        val = self.builder.bitcast(
            val,
            utils.get_llvm_type(context=self.context, type=types.voidptr),
        )
        self.builder.store(val, kernel_arg_dst)
        self.builder.store(
            numba_type_to_dpctl_typenum(self.context, ty), kernel_arg_ty_dst
        )

    def _build_complex_arg(
        self, val, ty, arg_list, args_ty_list, arg_num
    ):  # pylint: disable=too-many-arguments
        """Creates a list of LLVM Values for an unpacked complex kernel
        argument.
        """
        self._build_array_attr_arg(
            array_val=val,
            array_attr_pos=0,
            array_attr_ty=ty,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        self._build_array_attr_arg(
            array_val=val,
            array_attr_pos=1,
            array_attr_ty=ty,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1

    def _build_array_arg(  # pylint: disable=too-many-arguments
        self,
        array_val,
        array_data_model,
        arg_list,
        args_ty_list,
        arg_num,
    ):
        """Creates a list of LLVM Values for an unpacked DpnpNdArray kernel
        argument.

        The steps performed here are the same as in
        numba_dpex.core.kernel_interface.arg_pack_unpacker._unpack_array_helper
        """
        # Argument 1: Null pointer for the NRT_MemInfo attribute of the array
        nullptr = self._build_nullptr()
        self._build_arg(
            val=nullptr,
            ty=types.int64,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument 2: Null pointer for the Parent attribute of the array
        nullptr = self._build_nullptr()
        self._build_arg(
            val=nullptr,
            ty=types.int64,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument nitems
        self._build_array_attr_arg(
            array_val=array_val,
            array_attr_pos=array_data_model.get_field_position("nitems"),
            array_attr_ty=array_data_model.get_member_fe_type("nitems"),
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument itemsize
        self._build_array_attr_arg(
            array_val=array_val,
            array_attr_pos=array_data_model.get_field_position("itemsize"),
            array_attr_ty=array_data_model.get_member_fe_type("itemsize"),
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument data
        self._build_array_attr_arg(
            array_val=array_val,
            array_attr_pos=array_data_model.get_field_position("data"),
            array_attr_ty=array_data_model.get_member_fe_type("data"),
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument sycl_queue: as the queue pointer is not to be used in a
        # kernel we always pass in a nullptr
        self._build_arg(
            val=nullptr,
            ty=types.int64,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Arguments for shape
        shape_member = array_data_model.get_member_fe_type("shape")
        self._build_unituple_member_arg(
            array_val=array_val,
            array_attr_pos=array_data_model.get_field_position("shape"),
            ndims=shape_member.count,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += shape_member.count
        # Arguments for strides
        stride_member = array_data_model.get_member_fe_type("strides")
        self._build_unituple_member_arg(
            array_val=array_val,
            array_attr_pos=array_data_model.get_field_position("strides"),
            ndims=stride_member.count,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += stride_member.count

    # TODO: remove, not part of the builder
    def get_queue(self, exec_queue: dpctl.SyclQueue) -> llvmir.Instruction:
        """Allocates memory on the stack to store a DPCTLSyclQueueRef.

        Returns: A LLVM Value storing the pointer to the SYCL queue created
        using the filter string for the Python exec_queue (dpctl.SyclQueue).
        """

        # Allocate a stack var to store the queue created from the filter string
        sycl_queue_val = cgutils.alloca_once(
            self.builder,
            utils.get_llvm_type(context=self.context, type=types.voidptr),
        )
        # Insert a global constant to store the filter string
        device = self.context.insert_const_string(
            self.builder.module, exec_queue.sycl_device.filter_string
        )
        # Store the queue returned by DPEXRTQueue_CreateFromFilterString in a
        # local variable
        self.builder.store(
            self.rtctx.get_queue_from_filter_string(
                builder=self.builder, device=device
            ),
            sycl_queue_val,
        )
        return self.builder.load(sycl_queue_val)

    def _allocate_kernel_arg_array(self, num_kernel_args):
        """Allocates an array to store the LLVM Value for every kernel argument.

        Args:
            num_kernel_args (int): The number of kernel arguments that
            determines the size of args array to allocate.

        Returns: An LLVM IR value pointing to an array to store the kernel
        arguments.
        """
        args_list = cgutils.alloca_once(
            self.builder,
            utils.LLVMTypes.byte_ptr_t,
            size=self.context.get_constant(types.uintp, num_kernel_args),
        )

        return args_list

    def _allocate_kernel_arg_ty_array(self, num_kernel_args):
        """Allocates an array to store the LLVM Value for the typenum for
        every kernel argument.

        Args:
            num_kernel_args (int): The number of kernel arguments that
            determines the size of args array to allocate.

        Returns: An LLVM IR value pointing to an array to store the kernel
        arguments typenums as defined in dpctl.
        """
        args_ty_list = cgutils.alloca_once(
            self.builder,
            utils.LLVMTypes.int32_t,
            size=self.context.get_constant(types.uintp, num_kernel_args),
        )

        return args_ty_list

    def _create_sycl_range(self, idx_range):
        """Allocate a size_t[3] array to store the extents of a sycl::range.

        Sycl supports upto 3-dimensional ranges and a such the array is
        statically sized to length three. Only the elements that store an actual
        range value are populated based on the size of the idx_range argument.

        """
        intp_t = utils.get_llvm_type(context=self.context, type=types.intp)
        intp_ptr_t = utils.get_llvm_ptr_type(intp_t)
        num_dim = len(idx_range)

        # form the global range
        range_list = cgutils.alloca_once(
            self.builder,
            utils.get_llvm_type(context=self.context, type=types.uintp),
            size=self.context.get_constant(types.uintp, MAX_SIZE_OF_SYCL_RANGE),
        )

        for i in range(num_dim):
            rext = idx_range[i]
            if rext.type != utils.LLVMTypes.int64_t:
                rext = self.builder.sext(rext, utils.LLVMTypes.int64_t)

            # we reverse the global range to account for how sycl and opencl
            # range differs
            self.builder.store(
                rext,
                self.builder.gep(
                    range_list,
                    [self.context.get_constant(types.uintp, (num_dim - 1) - i)],
                ),
            )

        return self.builder.bitcast(range_list, intp_ptr_t)

    def set_kernel(self, sycl_kernel_ref: llvmir.Instruction):
        """Sets kernel to the argument list."""
        self.arguments.sycl_kernel_ref = sycl_kernel_ref

    def set_queue(self, sycl_queue_ref: llvmir.Instruction):
        """Sets queue to the argument list."""
        self.arguments.sycl_queue_ref = sycl_queue_ref

    def set_range(
        self,
        global_range: list,
        local_range: list = None,
    ):
        """Sets global and local range if provided to the argument list."""
        self.arguments.global_range = self._create_sycl_range(global_range)
        if local_range is not None and len(local_range) > 0:
            self.arguments.local_range = self._create_sycl_range(local_range)
        self.arguments.range_size = self.context.get_constant(
            types.uintp, len(global_range)
        )

    def set_arguments(
        self,
        ty_kernel_args: list,
        kernel_args: list,
    ):
        """Sets flattened kernel args, kernel arg types and number of those
        arguments to the argument list."""
        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self.builder,
                "DPEX-DEBUG: Populating kernel args and arg type arrays.\n",
            )

        num_flattened_kernel_args = self._get_num_flattened_kernel_args(
            kernel_argtys=ty_kernel_args,
        )

        # Create LLVM values for the kernel args list and kernel arg types list
        args_list = self._allocate_kernel_arg_array(num_flattened_kernel_args)

        args_ty_list = self._allocate_kernel_arg_ty_array(
            num_flattened_kernel_args
        )

        kernel_args_ptrs = []
        for arg in kernel_args:
            with self.builder.goto_entry_block():
                ptr = self.builder.alloca(arg.type)
            self.builder.store(arg, ptr)
            kernel_args_ptrs.append(ptr)

        # Populate the args_list and the args_ty_list LLVM arrays
        self._populate_kernel_args_and_args_ty_arrays(
            callargs_ptrs=kernel_args_ptrs,
            kernel_argtys=ty_kernel_args,
            args_list=args_list,
            args_ty_list=args_ty_list,
        )

        self.arguments.arg_list = args_list
        self.arguments.arg_ty_list = args_ty_list
        self.arguments.total_kernel_args = self.context.get_constant(
            types.uintp, num_flattened_kernel_args
        )

    def set_dependant_event_list(self, dep_events: list[llvmir.Instruction]):
        """Sets dependant events to the argument list."""
        if self.arguments.dep_events is not None:
            return

        if len(dep_events) > 0:
            # TODO: implement for non zero input
            raise NotImplementedError

        self.arguments.dep_events = self.builder.bitcast(
            utils.create_null_ptr(builder=self.builder, context=self.context),
            utils.get_llvm_type(context=self.context, type=types.voidptr),
        )
        self.arguments.dep_events_len = self.context.get_constant(
            types.uintp, 0
        )

    def submit(self) -> llvmir.Instruction:
        """Submits kernel by calling sycl.dpctl_queue_submit_range or
        sycl.dpctl_queue_submit_ndrange. Must be called after all arguments
        set."""
        args = self.arguments.to_list()

        if self.arguments.local_range is None:
            eref = sycl.dpctl_queue_submit_range(self.builder, *args)
        else:
            eref = sycl.dpctl_queue_submit_ndrange(self.builder, *args)

        return eref

    def _get_num_flattened_kernel_args(
        self,
        kernel_argtys: tuple[types.Type, ...],
    ) -> int:
        """Returns number of flattened arguments based on the numba types.
        flattens dpnp arrays and complex values."""
        num_flattened_kernel_args = 0
        for arg_type in kernel_argtys:
            if isinstance(arg_type, DpnpNdArray):
                datamodel = self.kernel_dmm.lookup(arg_type)
                num_flattened_kernel_args += datamodel.flattened_field_count
            elif arg_type in [types.complex64, types.complex128]:
                num_flattened_kernel_args += 2
            else:
                num_flattened_kernel_args += 1

        return num_flattened_kernel_args

    def _populate_kernel_args_and_args_ty_arrays(
        self,
        kernel_argtys,
        callargs_ptrs,
        args_list,
        args_ty_list,
    ):
        kernel_arg_num = 0
        for arg_num, argtype in enumerate(kernel_argtys):
            llvm_val = callargs_ptrs[arg_num]
            if isinstance(argtype, DpnpNdArray):
                datamodel = self.kernel_dmm.lookup(argtype)
                self._build_array_arg(
                    array_val=llvm_val,
                    array_data_model=datamodel,
                    arg_list=args_list,
                    args_ty_list=args_ty_list,
                    arg_num=kernel_arg_num,
                )
                kernel_arg_num += datamodel.flattened_field_count
            else:
                if argtype == types.complex64:
                    self._build_complex_arg(
                        llvm_val,
                        types.float32,
                        args_list,
                        args_ty_list,
                        kernel_arg_num,
                    )
                    kernel_arg_num += 2
                elif argtype == types.complex128:
                    self._build_complex_arg(
                        llvm_val,
                        types.float64,
                        args_list,
                        args_ty_list,
                        kernel_arg_num,
                    )
                    kernel_arg_num += 2
                else:
                    self._build_arg(
                        llvm_val,
                        argtype,
                        args_list,
                        args_ty_list,
                        kernel_arg_num,
                    )
                    kernel_arg_num += 1
