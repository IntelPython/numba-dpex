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

"""
This module provides a set of wrapper functions to insert dpctl C API function
declarations into an LLVM module.
"""

import llvmlite.llvmpy.core as lc
from llvmlite.ir import builder
from numba.core import types

import numba_dppy.utils as utils


class DpctlCAPIFnBuilder:
    """
    Defines a set of static functions to add declarations for dpctl C API
    library function into an LLVM module.
    """

    @staticmethod
    def _build_dpctl_function(builder, return_ty, arg_list, func_name):
        """
        _build_dpctl_function(builder, return_ty, arg_list, func_name)
        Inserts an LLVM function of the specified signature and name into an
        LLVM module.

        Args:
            return_ty: An LLVM Value corresponding to the return type of the
                       function.
            arg_list: A list of LLVM Value objects corresponsing to the
                      type of the arguments for the function.
            func_name: The name of the function passed as a string.

        Return: A Python object wrapping an LLVM Function.

        """
        func_ty = lc.Type.function(return_ty, arg_list)
        fn = builder.module.get_or_insert_function(func_ty, func_name)
        return fn

    @staticmethod
    def get_dpctl_queuemgr_get_current_queue(builder, context):
        """
        get_dpctl_queuemgr_get_current_queue(builder, context)
        Inserts an LLVM Function for ``DPCTLQueueMgr_GetCurrentQueue``.

        The ``DPCTLQueueMgr_GetCurrentQueue`` function returns the current
        top-of-the-stack SYCL queue that is stored in dpclt's queue manager.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Return: A Python object wrapping an LLVM Function for
                ``DPCTLQueueMgr_GetCurrentQueue``.
        """
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=utils.get_llvm_type(context=context, type=types.voidptr),
            arg_list=[],
            func_name="DPCTLQueueMgr_GetCurrentQueue",
        )

    @staticmethod
    def get_dpctl_queue_delete(builder, context):
        """
        get_dpctl_queue_delete(builder, context)
        Inserts an LLVM Function for the ``DPCTLQueue_Delete``.

        The ``DPCTLQueue_Delete`` deletes a DPCTLSyclQueueRef opaque pointer.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Return: A Python object wrapping an LLVM Function for
                ``DPCTLQueue_Delete``.
        """
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=utils.LLVMTypes.void_t,
            arg_list=[utils.get_llvm_type(context=context, type=types.voidptr)],
            func_name="DPCTLQueue_Delete",
        )

    @staticmethod
    def get_dpctl_queue_memcpy(builder, context):
        """
        get_dpctl_queue_memcpy(builder, context)
        Inserts an LLVM Function for the ``DPCTLQueue_Memcpy``.

        The ``DPCTLQueue_Memcpy`` function is a wrapper over ``sycl::queue``
        class' ``event memcpy(void* dest, const void* src, size_t numBytes)``
        function. Currently, the DPCTLQueue_Memcpy does not return an event
        reference, but in future will return an opaque pointer to a
        ``sycl::event``. All the opaque pointers arguments to the
        ``DPCTLQueue_Memcpy`` are passed as void pointers.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Return: A Python object wrapping an LLVM Function for
                ``DPCTLQueue_Memcpy``.
        """
        void_ptr_t = utils.get_llvm_type(context=context, type=types.voidptr)
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=utils.LLVMTypes.void_t,
            arg_list=[
                void_ptr_t,
                void_ptr_t,
                void_ptr_t,
                utils.get_llvm_type(context=context, type=types.intp),
            ],
            func_name="DPCTLQueue_Memcpy",
        )

    @staticmethod
    def get_dpctl_queue_wait(builder, context):
        """
        get_dpctl_queue_wait(builder, context)
        Inserts an LLVM Function for ``DPCTLQueue_Wait``.

        The ``DPCTLQueue_Wait`` is a wrapper over ``sycl::queue`` class'
        ``wait()`` function.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Returns: A Python object wrapping an LLVM Function for
                ``DPCTLQueue_Wait``.
        """
        void_ptr_t = utils.get_llvm_type(context=context, type=types.voidptr)
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=utils.LLVMTypes.void_t,
            arg_list=[void_ptr_t],
            func_name="DPCTLQueue_Wait",
        )

    @staticmethod
    def get_dpctl_queue_submit_range(builder, context):
        """
        get_dpctl_queue_submit_range(builder, context)
        Inserts an LLVM Function for ``DPCTLQueue_SubmitRange``.

        The ``DPCTLQueue_SubmitRange`` function is a wrapper over the
        ``sycl::queue`` class' ``event parallel_for(range<dimensions>
        numWorkItems, Rest&&... rest)`` function. All the opaque pointers
        arguments to the ``DPCTLQueue_SubmitRange`` function are passed as void
        pointers.

        Note: The ``DPCTLQueue_SubmitRange`` calls returns an opaque pointer to
        a ``sycl::event`` that needs to be destroyed properly.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Return: A Python object wrapping an LLVM Function for
                ``DPCTLQueue_SubmitRange``.
        """
        intp_t = utils.get_llvm_type(context=context, type=types.intp)
        void_ptr_t = utils.get_llvm_type(context=context, type=types.voidptr)
        void_ptr_ptr_t = utils.get_llvm_ptr_type(void_ptr_t)
        intp_ptr_t = utils.get_llvm_ptr_type(intp_t)

        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=void_ptr_t,
            arg_list=[
                void_ptr_t,
                void_ptr_t,
                void_ptr_ptr_t,
                utils.LLVMTypes.int32_ptr_t,
                utils.get_llvm_type(context=context, type=types.intp),
                intp_ptr_t,
                utils.get_llvm_type(context=context, type=types.intp),
                void_ptr_t,
                utils.get_llvm_type(context=context, type=types.intp),
            ],
            func_name="DPCTLQueue_SubmitRange",
        )

    @staticmethod
    def get_dpctl_malloc_shared(builder, context):
        """
        get_dpctl_malloc_shared(builder, context)
        Inserts an LLVM Function for ``DPCTLmalloc_shared``.

        ``DPCTLmalloc_shared`` is a wrapper over the
        ``sycl::malloc_shared`` function to allocate USM shared memory.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Return: A Python object wrapping an LLVM Function for
                ``DPCTLmalloc_shared``.

        """
        void_ptr_t = utils.get_llvm_type(context=context, type=types.voidptr)
        int_ptr_t = utils.get_llvm_type(context=context, type=types.intp)
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=void_ptr_t,
            arg_list=[int_ptr_t, void_ptr_t],
            func_name="DPCTLmalloc_shared",
        )

    @staticmethod
    def get_dpctl_free_with_queue(builder, context):
        """
        get_dpctl_free_with_queue(builder, context)
        Inserts an LLVM Function for ``DPCTLfree_with_queue``.

        The ``DPCTLfree_with_queue`` function is a wrapper over
        ``sycl::free(void*, queue)`` function. All the opaque pointers
        arguments to the ``DPCTLfree_with_queue`` are passed as void pointers.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Return: A Python object wrapping an LLVM Function for
                ``DPCTLfree_with_queue``.
        """
        void_ptr_t = utils.get_llvm_type(context=context, type=types.voidptr)
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=utils.LLVMTypes.void_t,
            arg_list=[void_ptr_t, void_ptr_t],
            func_name="DPCTLfree_with_queue",
        )

    @staticmethod
    def get_dpctl_event_wait(builder, context):
        """
        get_dpctl_event_wait(builder, context)
        Inserts an LLVM Function for ``DPCTLEvent_Wait``.

        The ``DPCTLEvent_Wait`` function is a wrapper over the
        ``sycl::event`` class' ``wait()`` function.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Return: A Python object wrapping an LLVM Function for
                ``DPCTLEvent_Wait``.
        """
        void_ptr_t = utils.get_llvm_type(context=context, type=types.voidptr)
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=utils.LLVMTypes.void_t,
            arg_list=[void_ptr_t],
            func_name="DPCTLEvent_Wait",
        )

    @staticmethod
    def get_dpctl_event_delete(builder, context):
        """
        get_dpctl_event_delete(builder, context)
        Inserts an LLVM Function for ``DPCTLEvent_Delete``.

        The ``DPCTLEvent_Delete`` function deletes a ``DPCTLSyclEventRef``
        opaque pointer.

        Args:
            builder: The LLVM IR builder to be used for code generation.
            context: The LLVM IR builder context.

        Returns: A Python object wrapping an LLVM Function for
                 ``DPCTLEvent_Delete``.
        """
        void_ptr_t = utils.get_llvm_type(context=context, type=types.voidptr)
        return DpctlCAPIFnBuilder._build_dpctl_function(
            builder,
            return_ty=utils.LLVMTypes.void_t,
            arg_list=[void_ptr_t],
            func_name="DPCTLEvent_Delete",
        )
