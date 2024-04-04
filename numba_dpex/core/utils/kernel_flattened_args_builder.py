# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides helpers to populate the list of kernel arguments that will
be passed to a DPCTLQueue_Submit function call by a KernelLaunchIRBuilder
object.
"""

from functools import reduce
from math import ceil
from typing import NamedTuple

import dpctl
from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.core.cpu import CPUContext

from numba_dpex.core.types import USMNdArray
from numba_dpex.core.types.kernel_api.local_accessor import (
    DpctlMDLocalAccessorType,
    LocalAccessorType,
)
from numba_dpex.core.utils import cgutils_extra as utils
from numba_dpex.dpctl_iface._helpers import numba_type_to_dpctl_typenum


class KernelArg(NamedTuple):
    """Stores the llvm IR value and the dpctl typeid for a kernel argument."""

    llvm_val: llvmir.Instruction
    typeid: int


class KernelFlattenedArgsBuilder:
    """Helper to generate the flattened list of kernel arguments to be
    passed to a DPCTLQueue_Submit function.

    **Note** Two separate data models are used when building a flattened
    kernel argument for the following reason:

    Different numba-dpex targets can use different data models for the same
    data type that may have different number of attributes and a different
    type for each attribute.

    In the case the DpnpNdArray type, two separate data models are used for
    the CPUTarget and for the SPIRVTarget. The SPIRVTarget does not have the
    ``parent``, ``meminfo`` and ``sycl_queue`` attributes that are present
    in the data model used by the CPUTarget. The SPIRVTarget's data model
    for DpnpNdArray also requires an explicit address space qualifier for
    the ``data`` attribute.

    When generating the LLVM IR for the host-side control code for executing
    a SPIR-V kernel, the kernel arguments are represented using the
    CPUTarget's data model for each argument's type. However, the actual
    kernel function generated as a SPIR-V binary by the SPIRVTarget uses its
    own data model manager to build the flattened kernel function argument
    list. For this reason, when building the flattened argument list for a
    kernel launch call the host data model is used to extract the
    required attributes and then the kernel data model is used to get the
    correct type for the attribute.
    """

    def __init__(
        self,
        context: CPUContext,
        builder: llvmir.IRBuilder,
        kernel_dmm,
    ):
        self._context = context
        self._builder = builder
        self._kernel_dmm = kernel_dmm
        self._kernel_arg_list = []

    def add_argument(
        self,
        arg_type,
        arg_packed_llvm_val,
    ):
        """Add flattened representation of a kernel argument."""
        if isinstance(arg_type, LocalAccessorType):
            self._kernel_arg_list.extend(
                self._build_local_accessor_arg(
                    arg_type, llvm_val=arg_packed_llvm_val
                )
            )
        elif isinstance(arg_type, USMNdArray):
            self._kernel_arg_list.extend(
                self._build_array_arg(
                    arg_type, llvm_array_val=arg_packed_llvm_val
                )
            )
        elif arg_type == types.complex64:
            self._kernel_arg_list.extend(
                self._build_complex_arg(
                    llvm_val=arg_packed_llvm_val,
                    numba_type=types.float32,
                )
            )
        elif arg_type == types.complex128:
            self._kernel_arg_list.extend(
                self._build_complex_arg(
                    llvm_val=arg_packed_llvm_val,
                    numba_type=types.float64,
                )
            )
        else:
            self._kernel_arg_list.extend(
                self._build_arg(
                    llvm_val=arg_packed_llvm_val, numba_type=arg_type
                )
            )

    def get_kernel_arg_list(self) -> list[KernelArg]:
        """Returns a list of KernelArg objects representing a flattened kernel
        argument.

        Returns:
            list[KernelArg]: List of flattened KernelArg objects
        """
        return self._kernel_arg_list

    def print_kernel_arg_list(self) -> None:
        """Prints out the kernel argument list in a human readable format.

        Args:
            args_list (list[KernelArg]): List of kernel arguments to be printed
        """
        args_list = self._kernel_arg_list
        print(f"Number of flattened kernel arguments: {len(args_list)}")
        for karg in args_list:
            print(f"    {karg.llvm_val} of typeid {karg.typeid}")

    def _build_arg(self, llvm_val, numba_type):
        """Returns a KernelArg to be passed to a DPCTLQueue_Submit call.

        The passed in LLVM IR Value is bitcast to a void* and the
        numba/numba_dpex type object is mapped to the corresponding
        DPCTLKernelArgType enum value and returned back as a KernelArg object.

        Args:
            llvm_val: An LLVM IR Value that will be stored into the arguments
                array
            numba_type: A Numba type that will be converted to a
                DPCTLKernelArgType enum and stored into the argument types
                list array
        Returns:
            KernelArg: Tuple corresponding to the LLVM IR Instruction and
                DPCTLKernelArgType enum value.
        """
        llvm_val = self._builder.bitcast(
            llvm_val,
            utils.get_llvm_type(context=self._context, type=types.voidptr),
        )
        typeid = numba_type_to_dpctl_typenum(self._context, numba_type)

        return [KernelArg(llvm_val, typeid)]

    def _build_unituple_member_arg(self, llvm_val, attr_pos, ndims):
        kernel_arg_list = []
        array_attr = self._builder.gep(
            llvm_val,
            [
                self._context.get_constant(types.int32, 0),
                self._context.get_constant(types.int32, attr_pos),
            ],
        )

        for ndim in range(ndims):
            kernel_arg_list.extend(
                self._build_collections_attr_arg(
                    llvm_val=array_attr,
                    attr_index=ndim,
                    attr_type=types.int64,
                )
            )

        return kernel_arg_list

    def _build_collections_attr_arg(self, llvm_val, attr_index, attr_type):
        array_attr = self._builder.gep(
            llvm_val,
            [
                self._context.get_constant(types.int32, 0),
                self._context.get_constant(types.int32, attr_index),
            ],
        )

        if isinstance(attr_type, (types.misc.RawPointer, types.misc.CPointer)):
            array_attr = self._builder.load(array_attr)

        return self._build_arg(llvm_val=array_attr, numba_type=attr_type)

    def _build_complex_arg(self, llvm_val, numba_type):
        """Creates a list of LLVM Values for an unpacked complex kernel
        argument.
        """
        kernel_arg_list = []

        kernel_arg_list.extend(
            self._build_collections_attr_arg(
                llvm_val=llvm_val,
                attr_index=0,
                attr_type=numba_type,
            )
        )
        kernel_arg_list.extend(
            self._build_collections_attr_arg(
                llvm_val=llvm_val,
                attr_index=1,
                attr_type=numba_type,
            )
        )

        return kernel_arg_list

    def _store_val_into_struct(self, struct_ref, index, val):
        self._builder.store(
            val,
            self._builder.gep(
                struct_ref,
                [
                    self._context.get_constant(types.int32, 0),
                    self._context.get_constant(types.int32, index),
                ],
            ),
        )

    def _build_local_accessor_metadata_arg(
        self, llvm_val, arg_type: LocalAccessorType, data_attr_ty
    ):
        """Handles the special case of building the kernel argument for the data
        attribute of a kernel_api.LocalAccessor object.

        A kernel_api.LocalAccessor conceptually represents a device-only memory
        allocation. The mock kernel_api.LocalAccessor uses a numpy.ndarray to
        represent the data allocation. The numpy.ndarray cannot be passed to the
        kernel and is ignored when building the kernel argument. Instead, a
        struct is allocated to store the metadata about the size of the device
        memory allocation and a reference to the struct is passed to the
        DPCTLQueue_Submit call. The DPCTLQueue_Submit then constructs a
        sycl::local_accessor object using the metadata and passes the
        sycl::local_accessor as the kernel argument, letting the DPC++ runtime
        handle proper device memory allocation.
        """

        ndim = arg_type.ndim

        md_proxy = cgutils.create_struct_proxy(DpctlMDLocalAccessorType())(
            self._context,
            self._builder,
        )
        la_proxy = cgutils.create_struct_proxy(arg_type)(
            self._context, self._builder, value=self._builder.load(llvm_val)
        )

        md_proxy.ndim = self._context.get_constant(types.int64, ndim)
        md_proxy.dpctl_type_id = numba_type_to_dpctl_typenum(
            self._context, data_attr_ty.dtype
        )
        for i, val in enumerate(
            cgutils.unpack_tuple(self._builder, la_proxy.shape)
        ):
            setattr(md_proxy, f"dim{i}", val)

        return self._build_arg(
            llvm_val=md_proxy._getpointer(),
            numba_type=LocalAccessorType(
                ndim, dpctl.tensor.dtype(data_attr_ty.dtype.name)
            ),
        )

    def _build_local_accessor_arg(self, arg_type: LocalAccessorType, llvm_val):
        """Creates a list of kernel LLVM Values for an unpacked USMNdArray
        kernel argument from the local accessor.

        Method generates UsmNdArray fields from local accessor type and value.
        """
        # TODO: move extra values build on device side of codegen.
        ndim = arg_type.ndim
        la_proxy = cgutils.create_struct_proxy(arg_type)(
            self._context, self._builder, value=self._builder.load(llvm_val)
        )
        shape = cgutils.unpack_tuple(self._builder, la_proxy.shape)
        ll_size = reduce(self._builder.mul, shape)

        size_ptr = cgutils.alloca_once_value(self._builder, ll_size)
        itemsize = self._context.get_constant(
            types.intp, ceil(arg_type.dtype.bitwidth / types.byte.bitwidth)
        )
        itemsize_ptr = cgutils.alloca_once_value(self._builder, itemsize)

        kernel_arg_list = []

        kernel_dm = self._kernel_dmm.lookup(arg_type)

        kernel_arg_list.extend(
            self._build_arg(
                llvm_val=size_ptr,
                numba_type=kernel_dm.get_member_fe_type("nitems"),
            )
        )

        # Argument itemsize
        kernel_arg_list.extend(
            self._build_arg(
                llvm_val=itemsize_ptr,
                numba_type=kernel_dm.get_member_fe_type("itemsize"),
            )
        )

        # Argument data
        data_attr_ty = kernel_dm.get_member_fe_type("data")

        kernel_arg_list.extend(
            self._build_local_accessor_metadata_arg(
                llvm_val=llvm_val,
                arg_type=arg_type,
                data_attr_ty=data_attr_ty,
            )
        )

        # Arguments for shape
        for val in shape:
            shape_ptr = cgutils.alloca_once_value(self._builder, val)
            kernel_arg_list.extend(
                self._build_arg(
                    llvm_val=shape_ptr,
                    numba_type=types.int64,
                )
            )

        # Arguments for strides
        for i in range(ndim):
            kernel_arg_list.extend(
                self._build_arg(
                    llvm_val=itemsize_ptr,
                    numba_type=types.int64,
                )
            )

        return kernel_arg_list

    def _build_array_arg(self, arg_type, llvm_array_val):
        """Creates a list of LLVM Values for an unpacked USMNdArray kernel
        argument.
        """
        kernel_arg_list = []

        kernel_data_model = self._kernel_dmm.lookup(arg_type)
        host_data_model = self._context.data_model_manager.lookup(arg_type)

        kernel_arg_list.extend(
            self._build_collections_attr_arg(
                llvm_val=llvm_array_val,
                attr_index=host_data_model.get_field_position("nitems"),
                attr_type=kernel_data_model.get_member_fe_type("nitems"),
            )
        )
        # Argument itemsize
        kernel_arg_list.extend(
            self._build_collections_attr_arg(
                llvm_val=llvm_array_val,
                attr_index=host_data_model.get_field_position("itemsize"),
                attr_type=kernel_data_model.get_member_fe_type("itemsize"),
            )
        )
        # Argument data
        data_attr_pos = host_data_model.get_field_position("data")
        data_attr_ty = kernel_data_model.get_member_fe_type("data")

        kernel_arg_list.extend(
            self._build_collections_attr_arg(
                llvm_val=llvm_array_val,
                attr_index=data_attr_pos,
                attr_type=data_attr_ty,
            )
        )
        # Arguments for shape
        kernel_arg_list.extend(
            self._build_unituple_member_arg(
                llvm_val=llvm_array_val,
                attr_pos=host_data_model.get_field_position("shape"),
                ndims=kernel_data_model.get_member_fe_type("shape").count,
            )
        )
        # Arguments for strides
        kernel_arg_list.extend(
            self._build_unituple_member_arg(
                llvm_val=llvm_array_val,
                attr_pos=host_data_model.get_field_position("strides"),
                ndims=kernel_data_model.get_member_fe_type("strides").count,
            )
        )

        return kernel_arg_list
