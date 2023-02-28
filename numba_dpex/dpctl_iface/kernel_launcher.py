# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import cgutils, types

from numba_dpex import utils
from numba_dpex.core.runtime.context import DpexRTContext
from numba_dpex.dpctl_iface import DpctlCAPIFnBuilder
from numba_dpex.dpctl_iface._helpers import numba_type_to_dpctl_typenum


class KernelLaunchIRBuilder:
    """
    KernelLaunchIRBuilder(lowerer, cres)
    Helper class to build the LLVM IR for the submission of a kernel.

    The class generates LLVM IR inside the current LLVM module that is needed
    for submitting kernels. The LLVM Values that
    """

    def __init__(self, lowerer, kernel):
        """Create a KernelLauncher for the specified kernel.

        Args:
            lowerer: The Numba Lowerer that will be used to generate the code.
            kernel: The SYCL kernel for which we are generating the code.
            num_inputs: The number of arguments to the kernels.
        """
        self.lowerer = lowerer
        self.context = self.lowerer.context
        self.builder = self.lowerer.builder
        self.kernel = kernel
        self.kernel_addr = self.kernel.addressof_ref()
        self.rtctx = DpexRTContext(self.context)

    def _build_nullptr(self):
        """Builds the LLVM IR to represent a null pointer.

        Returns: An LLVM Value storing a null pointer
        """
        zero = cgutils.alloca_once(self.builder, utils.LLVMTypes.int64_t)
        self.builder.store(self.context.get_constant(types.int64, 0), zero)
        return self.builder.bitcast(
            zero, utils.get_llvm_type(context=self.context, type=types.voidptr)
        )

    def _build_array_attr_arg(
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
        if isinstance(array_attr_ty, types.misc.RawPointer):
            array_attr = self.builder.load(array_attr)

        self.build_arg(
            val=array_attr,
            ty=array_attr_ty,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )

    def _build_flattened_array_args(
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

    def build_arg(self, val, ty, arg_list, args_ty_list, arg_num):
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

    def build_array_arg(
        self, array_val, array_rank, arg_list, args_ty_list, arg_num
    ):
        """Creates a list of LLVM Values for an unpacked DpnpNdArray kernel
        argument.

        The steps performed here are the same as in
        numba_dpex.core.kernel_interface.arg_pack_unpacker._unpack_array_helper
        """
        # Argument 1: Null pointer for the NRT_MemInfo attribute of the array
        nullptr = self._build_nullptr()
        self.build_arg(
            val=nullptr,
            ty=types.int64,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument 2: Null pointer for the Parent attribute of the array
        nullptr = self._build_nullptr()
        self.build_arg(
            val=nullptr,
            ty=types.int64,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument 3: Array size
        self._build_array_attr_arg(
            array_val=array_val,
            array_attr_pos=2,
            array_attr_ty=types.int64,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument 4: itemsize
        self._build_array_attr_arg(
            array_val=array_val,
            array_attr_pos=3,
            array_attr_ty=types.int64,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Argument 5: data pointer
        self._build_array_attr_arg(
            array_val=array_val,
            array_attr_pos=4,
            array_attr_ty=types.voidptr,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += 1
        # Arguments for flattened shape
        self._build_flattened_array_args(
            array_val=array_val,
            array_attr_pos=5,
            ndims=array_rank,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += array_rank
        # Arguments for flattened stride
        self._build_flattened_array_args(
            array_val=array_val,
            array_attr_pos=6,
            ndims=array_rank,
            arg_list=arg_list,
            args_ty_list=args_ty_list,
            arg_num=arg_num,
        )
        arg_num += array_rank

    def get_queue(self, exec_queue):
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
        return sycl_queue_val

    def free_queue(self, sycl_queue_val):
        """
        Frees the ``DPCTLSyclQueueRef`` pointer that was used to launch the
        kernels.

        Args:
            sycl_queue_val: The SYCL queue pointer to be freed.
        """
        fn = DpctlCAPIFnBuilder.get_dpctl_queue_delete(
            builder=self.builder, context=self.context
        )
        self.builder.call(fn, [self.builder.load(sycl_queue_val)])

    def allocate_kernel_arg_array(self, num_kernel_args):
        """Allocates an array to store the LLVM Value for every kernel argument.

        Args:
            num_kernel_args (int): The number of kernel arguments that
            determines the size of args array to allocate.

        Returns: An LLVM IR value pointing to an array to store the kernel
        arguments.
        """
        args_list = cgutils.alloca_once(
            self.builder,
            utils.get_llvm_type(context=self.context, type=types.voidptr),
            size=self.context.get_constant(types.uintp, num_kernel_args),
        )

        return args_list

    def allocate_kernel_arg_ty_array(self, num_kernel_args):
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
        """_summary_

        Args:
            idx_range (_type_): _description_
            kernel_name_tag (_type_): _description_
        """
        intp_t = utils.get_llvm_type(context=self.context, type=types.intp)
        intp_ptr_t = utils.get_llvm_ptr_type(intp_t)
        num_dim = len(idx_range)

        # form the global range
        global_range = cgutils.alloca_once(
            self.builder,
            utils.get_llvm_type(context=self.context, type=types.uintp),
            size=self.context.get_constant(types.uintp, num_dim),
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
                    global_range,
                    [self.context.get_constant(types.uintp, (num_dim - 1) - i)],
                ),
            )

        return self.builder.bitcast(global_range, intp_ptr_t)

    def submit_sync_ranged_kernel(
        self,
        idx_range,
        sycl_queue_val,
        total_kernel_args,
        arg_list,
        arg_ty_list,
    ):
        """
        submit_sync_ranged_kernel(dim_bounds, sycl_queue_val)
        Submits the kernel to the specified queue, waits and then copies
        back any results to the host.

        Args:
            idx_range: Tuple specifying the range over which the kernel is
            to be submitted.
            sycl_queue_val : The SYCL queue on which the kernel is
                             submitted.
        """
        gr = self._create_sycl_range(idx_range)
        args = [
            self.builder.inttoptr(
                self.context.get_constant(types.uintp, self.kernel_addr),
                utils.get_llvm_type(context=self.context, type=types.voidptr),
            ),
            self.builder.load(sycl_queue_val),
            arg_list,
            arg_ty_list,
            self.context.get_constant(types.uintp, total_kernel_args),
            gr,
            self.context.get_constant(types.uintp, len(idx_range)),
            self.builder.bitcast(
                utils.create_null_ptr(
                    builder=self.builder, context=self.context
                ),
                utils.get_llvm_type(context=self.context, type=types.voidptr),
            ),
            self.context.get_constant(types.uintp, 0),
        ]

        self.rtctx.submit_range(self.builder, *args)
