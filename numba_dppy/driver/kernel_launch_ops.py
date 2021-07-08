# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numba.core import cgutils, types
from numba.core.ir_utils import legalize_names

from numba_dppy import numpy_usm_shared as nus
from numba_dppy import utils
from numba_dppy.driver import DpctlCAPIFnBuilder
from numba_dppy.driver._helpers import numba_type_to_dpctl_typenum


class KernelLaunchOps:
    """
    KernelLaunchOps(lowerer, cres, num_inputs)
    Defines a set of functions to launch a SYCL kernel on the "current queue"
    as defined in the dpctl queue manager.
    """

    def _form_kernel_arg_and_arg_ty(self, val, ty):
        kernel_arg_dst = self.builder.gep(
            self.kernel_arg_array,
            [self.context.get_constant(types.int32, self.cur_arg)],
        )
        kernel_arg_ty_dst = self.builder.gep(
            self.kernel_arg_ty_array,
            [self.context.get_constant(types.int32, self.cur_arg)],
        )
        self.cur_arg += 1
        self.builder.store(val, kernel_arg_dst)
        self.builder.store(ty, kernel_arg_ty_dst)

    def __init__(self, lowerer, kernel, num_inputs):
        """Create an instance of KernelLaunchOps.

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
        self.total_kernel_args = 0
        self.cur_arg = 0
        self.num_inputs = num_inputs
        # list of buffer that needs to comeback to host
        self.write_buffs = []
        # list of buffer that does not need to comeback to host
        self.read_only_buffs = []

    def get_current_queue(self):
        """Allocates memory on the stack to store the current queue from dpctl.

        A SYCL queue is needed to allocate USM memory and submit a kernel. This
        function gets the queue returned by ``DPCTLQueueMgr_GetCurrentQueue``
        function and stores it on the stack. The queue should be freed properly
        after returning from the kernel.

        Return: A LLVM Value storing the pointer to the SYCL queue returned
                by ``DPCTLQueueMgr_GetCurrentQueue``.

        """
        sycl_queue_val = cgutils.alloca_once(
            self.builder, utils.get_llvm_type(context=self.context, type=types.voidptr)
        )
        fn = DpctlCAPIFnBuilder.get_dpctl_queuemgr_get_current_queue(
            builder=self.builder, context=self.context
        )
        self.builder.store(self.builder.call(fn, []), sycl_queue_val)
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
            num_kernel_args : The number of kernel arguments for the kernel.

        """

        self.total_kernel_args = num_kernel_args

        # we need a kernel arg array to enqueue
        self.kernel_arg_array = cgutils.alloca_once(
            self.builder,
            utils.get_llvm_type(context=self.context, type=types.voidptr),
            size=self.context.get_constant(types.uintp, num_kernel_args),
            name="kernel_arg_array",
        )

        self.kernel_arg_ty_array = cgutils.alloca_once(
            self.builder,
            utils.LLVMTypes.int32_t,
            size=self.context.get_constant(types.uintp, num_kernel_args),
            name="kernel_arg_ty_array",
        )

    def process_kernel_arg(
        self, var, llvm_arg, arg_type, index, modified_arrays, sycl_queue_val
    ):
        """
        process_kernel_arg(var, llvm_arg, arg_type, index, modified_arrays,
        sycl_queue_val)
        Creates an LLVM Value for each kernel argument.

        Args:
            var : A kernel argument represented as a Numba type.
            llvm_arg : Only used for array arguments and points to the LLVM
                       value previously allocated to store the array arg.
            arg_type : The Numba type for the argument.
            index : The poisition of the argument in the list of arguments.
            modified_arrays : The list of array arguments that are written to
                              inside the kernel. The list is used to check if
                              the argument is read-only or not.

        Raises:
            NotImplementedError: If an unsupported type of kernel argument is
                                 encountered.
        """
        if isinstance(arg_type, types.npytypes.Array):
            if llvm_arg is None:
                raise NotImplementedError(arg_type, var)

            storage = cgutils.alloca_once(self.builder, utils.LLVMTypes.int64_t)
            self.builder.store(self.context.get_constant(types.int64, 0), storage)
            ty = numba_type_to_dpctl_typenum(context=self.context, type=types.int64)
            self._form_kernel_arg_and_arg_ty(
                self.builder.bitcast(
                    storage,
                    utils.get_llvm_type(context=self.context, type=types.voidptr),
                ),
                ty,
            )

            storage = cgutils.alloca_once(self.builder, utils.LLVMTypes.int64_t)
            self.builder.store(self.context.get_constant(types.int64, 0), storage)
            ty = numba_type_to_dpctl_typenum(context=self.context, type=types.int64)
            self._form_kernel_arg_and_arg_ty(
                self.builder.bitcast(
                    storage,
                    utils.get_llvm_type(context=self.context, type=types.voidptr),
                ),
                ty,
            )

            # Handle array size
            array_size_member = self.builder.gep(
                llvm_arg,
                [
                    self.context.get_constant(types.int32, 0),
                    self.context.get_constant(types.int32, 2),
                ],
            )

            ty = numba_type_to_dpctl_typenum(context=self.context, type=types.int64)
            self._form_kernel_arg_and_arg_ty(
                self.builder.bitcast(
                    array_size_member,
                    utils.get_llvm_type(context=self.context, type=types.voidptr),
                ),
                ty,
            )

            # Handle itemsize
            item_size_member = self.builder.gep(
                llvm_arg,
                [
                    self.context.get_constant(types.int32, 0),
                    self.context.get_constant(types.int32, 3),
                ],
            )

            ty = numba_type_to_dpctl_typenum(context=self.context, type=types.int64)
            self._form_kernel_arg_and_arg_ty(
                self.builder.bitcast(
                    item_size_member,
                    utils.get_llvm_type(context=self.context, type=types.voidptr),
                ),
                ty,
            )

            # Calculate total buffer size
            total_size = cgutils.alloca_once(
                self.builder,
                utils.get_llvm_type(context=self.context, type=types.intp),
                size=utils.get_one(context=self.context),
                name="total_size" + str(self.cur_arg),
            )
            self.builder.store(
                self.builder.sext(
                    self.builder.mul(
                        self.builder.load(array_size_member),
                        self.builder.load(item_size_member),
                    ),
                    utils.get_llvm_type(context=self.context, type=types.intp),
                ),
                total_size,
            )

            # Handle data
            data_member = self.builder.gep(
                llvm_arg,
                [
                    self.context.get_constant(types.int32, 0),
                    self.context.get_constant(types.int32, 4),
                ],
            )

            # names are replaced using legalize names, we have to do the same
            # here for them to match.
            legal_names = legalize_names([var])
            ty = numba_type_to_dpctl_typenum(context=self.context, type=types.voidptr)

            if isinstance(arg_type, nus.UsmSharedArrayType):
                self._form_kernel_arg_and_arg_ty(
                    self.builder.bitcast(
                        self.builder.load(data_member),
                        utils.get_llvm_type(context=self.context, type=types.voidptr),
                    ),
                    ty,
                )
            else:
                malloc_fn = DpctlCAPIFnBuilder.get_dpctl_malloc_shared(
                    builder=self.builder, context=self.context
                )
                memcpy_fn = DpctlCAPIFnBuilder.get_dpctl_queue_memcpy(
                    builder=self.builder, context=self.context
                )

                # Not known to be USM so we need to copy to USM.
                buffer_name = "buffer_ptr" + str(self.cur_arg)
                # Create void * to hold new USM buffer.
                buffer_ptr = cgutils.alloca_once(
                    self.builder,
                    utils.get_llvm_type(context=self.context, type=types.voidptr),
                    name=buffer_name,
                )
                # Setup the args to the USM allocator, size and SYCL queue.
                args = [
                    self.builder.load(total_size),
                    self.builder.load(sycl_queue_val),
                ]
                # Call USM shared allocator and store in buffer_ptr.
                self.builder.store(self.builder.call(malloc_fn, args), buffer_ptr)

                if legal_names[var] in modified_arrays:
                    self.write_buffs.append((buffer_ptr, total_size, data_member))
                else:
                    self.read_only_buffs.append((buffer_ptr, total_size, data_member))

                # We really need to detect when an array needs to be copied over
                if index < self.num_inputs:
                    args = [
                        self.builder.load(sycl_queue_val),
                        self.builder.load(buffer_ptr),
                        self.builder.bitcast(
                            self.builder.load(data_member),
                            utils.get_llvm_type(
                                context=self.context, type=types.voidptr
                            ),
                        ),
                        self.builder.load(total_size),
                    ]
                    self.builder.call(memcpy_fn, args)

                self._form_kernel_arg_and_arg_ty(self.builder.load(buffer_ptr), ty)

            # Handle shape
            shape_member = self.builder.gep(
                llvm_arg,
                [
                    self.context.get_constant(types.int32, 0),
                    self.context.get_constant(types.int32, 5),
                ],
            )

            for this_dim in range(arg_type.ndim):
                shape_entry = self.builder.gep(
                    shape_member,
                    [
                        self.context.get_constant(types.int32, 0),
                        self.context.get_constant(types.int32, this_dim),
                    ],
                )
                ty = numba_type_to_dpctl_typenum(context=self.context, type=types.int64)
                self._form_kernel_arg_and_arg_ty(
                    self.builder.bitcast(
                        shape_entry,
                        utils.get_llvm_type(context=self.context, type=types.voidptr),
                    ),
                    ty,
                )

            # Handle strides
            stride_member = self.builder.gep(
                llvm_arg,
                [
                    self.context.get_constant(types.int32, 0),
                    self.context.get_constant(types.int32, 6),
                ],
            )

            for this_stride in range(arg_type.ndim):
                stride_entry = self.builder.gep(
                    stride_member,
                    [
                        self.context.get_constant(types.int32, 0),
                        self.context.get_constant(types.int32, this_stride),
                    ],
                )

                ty = numba_type_to_dpctl_typenum(context=self.context, type=types.int64)
                self._form_kernel_arg_and_arg_ty(
                    self.builder.bitcast(
                        stride_entry,
                        utils.get_llvm_type(context=self.context, type=types.voidptr),
                    ),
                    ty,
                )

        else:
            ty = numba_type_to_dpctl_typenum(context=self.context, type=arg_type)
            self._form_kernel_arg_and_arg_ty(
                self.builder.bitcast(
                    llvm_arg,
                    utils.get_llvm_type(context=self.context, type=types.voidptr),
                ),
                ty,
            )

    def enqueue_kernel_and_copy_back(self, dim_bounds, sycl_queue_val):
        """
        enqueue_kernel_and_copy_back(dim_bounds, sycl_queue_val)
        Submits the kernel to the specified queue, waits and then copies
        back any results to the host.

        Args:
            dim_bounds : An array of three tuple representing the starting
                         offset, end offset and the stride (step) for each
                         dimension of the input arrays. Every array in a parfor
                         is of the same dimensionality and shape, thus ensuring
                         the bounds are the same.
            sycl_queue_val : The SYCL queue on which the kernel is
                             submitted.
        """
        submit_fn = DpctlCAPIFnBuilder.get_dpctl_queue_submit_range(
            builder=self.builder, context=self.context
        )
        queue_wait_fn = DpctlCAPIFnBuilder.get_dpctl_queue_wait(
            builder=self.builder, context=self.context
        )
        event_del_fn = DpctlCAPIFnBuilder.get_dpctl_event_delete(
            builder=self.builder, context=self.context
        )
        memcpy_fn = DpctlCAPIFnBuilder.get_dpctl_queue_memcpy(
            builder=self.builder, context=self.context
        )
        free_fn = DpctlCAPIFnBuilder.get_dpctl_free_with_queue(
            builder=self.builder, context=self.context
        )

        # the assumption is loop_ranges will always be less than or equal to 3
        # dimensions
        num_dim = len(dim_bounds) if len(dim_bounds) < 4 else 3

        # form the global range
        global_range = cgutils.alloca_once(
            self.builder,
            utils.get_llvm_type(context=self.context, type=types.uintp),
            size=self.context.get_constant(types.uintp, num_dim),
            name="global_range",
        )

        intp_t = utils.get_llvm_type(context=self.context, type=types.intp)
        intp_ptr_t = utils.get_llvm_ptr_type(intp_t)

        for i in range(num_dim):
            start, stop, step = dim_bounds[i]
            if stop.type != utils.LLVMTypes.int64_t:
                stop = self.builder.sext(stop, utils.LLVMTypes.int64_t)

            # we reverse the global range to account for how sycl and opencl
            # range differs
            self.builder.store(
                stop,
                self.builder.gep(
                    global_range,
                    [self.context.get_constant(types.uintp, (num_dim - 1) - i)],
                ),
            )

        args = [
            self.builder.inttoptr(
                self.context.get_constant(types.uintp, self.kernel_addr),
                utils.get_llvm_type(context=self.context, type=types.voidptr),
            ),
            self.builder.load(sycl_queue_val),
            self.kernel_arg_array,
            self.kernel_arg_ty_array,
            self.context.get_constant(types.uintp, self.total_kernel_args),
            self.builder.bitcast(global_range, intp_ptr_t),
            self.context.get_constant(types.uintp, num_dim),
            self.builder.bitcast(
                utils.create_null_ptr(builder=self.builder, context=self.context),
                utils.get_llvm_type(context=self.context, type=types.voidptr),
            ),
            self.context.get_constant(types.uintp, 0),
        ]
        # Submit the kernel
        event_ref = self.builder.call(submit_fn, args)
        # Add a wait on the queue
        self.builder.call(queue_wait_fn, [self.builder.load(sycl_queue_val)])
        # Note that the dpctl_queue_wait call waits on the event and then
        # decrements the ref count of the sycl::event C++ object. However, the
        # event object returned by the get_dpctl_queue_submit_range call still
        # needs to be explicitly deleted to free up the event object properly.
        self.builder.call(event_del_fn, [event_ref])

        # read buffers back to host
        for write_buff in self.write_buffs:
            buffer_ptr, total_size, data_member = write_buff
            args = [
                self.builder.load(sycl_queue_val),
                self.builder.bitcast(
                    self.builder.load(data_member),
                    utils.get_llvm_type(context=self.context, type=types.voidptr),
                ),
                self.builder.load(buffer_ptr),
                self.builder.load(total_size),
            ]
            # FIXME: In future, when the DctlQueue_Memcpy is made non-blocking
            # the returned event should be explicitly freed by calling
            # get_dpctl_event_delete.
            self.builder.call(memcpy_fn, args)

            self.builder.call(
                free_fn,
                [
                    self.builder.load(buffer_ptr),
                    self.builder.load(sycl_queue_val),
                ],
            )

        for read_buff in self.read_only_buffs:
            buffer_ptr, total_size, data_member = read_buff
            self.builder.call(
                free_fn,
                [
                    self.builder.load(buffer_ptr),
                    self.builder.load(sycl_queue_val),
                ],
            )
