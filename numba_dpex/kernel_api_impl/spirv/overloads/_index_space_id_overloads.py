# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements the SPIR-V overloads for the kernel_api.items class methods.
"""

import llvmlite.ir as llvmir
from numba.core import cgutils, types
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload_attribute, overload_method

from numba_dpex.core.types.kernel_api.index_space_ids import (
    GroupType,
    ItemType,
    NdItemType,
)
from numba_dpex.kernel_api import Group, Item, NdItem
from numba_dpex.kernel_api_impl.spirv.target import SPIRVTargetContext

from ..target import SPIRV_TARGET_NAME


def spirv_name(name: str):
    """Converts name to spirv name by adding __spirv_ prefix."""
    return "__spirv_" + name


def declare_spirv_const(
    builder: llvmir.IRBuilder,
    name: str,
):
    """Declares global external spirv constant"""
    data = cgutils.add_global_variable(
        builder.module,
        llvmir.VectorType(llvmir.IntType(64), 3),
        spirv_name(name),
        addrspace=1,
    )
    data.linkage = "external"
    data.global_constant = True
    data.align = 32
    data.storage_class = "dso_local local_unnamed_addr"
    return data


def _intrinsic_spirv_global_index_const(
    ty_context,  # pylint: disable=unused-argument
    ty_dim,  # pylint: disable=unused-argument
    const_name: str,
):
    """Generates instruction to get spirv index from const_name."""
    sig = types.int64(types.int32)

    def _intrinsic_spirv_global_index_const_gen(
        context: SPIRVTargetContext,  # pylint: disable=unused-argument
        builder: llvmir.IRBuilder,
        sig,  # pylint: disable=unused-argument
        args,
    ):
        index_const = declare_spirv_const(
            builder,
            const_name,
        )
        [dim] = args
        # TODO: llvmlite does not support gep on vector. Use this instead once
        # supported.
        # https://github.com/numba/llvmlite/issues/756
        # res = builder.gep( # noqa: E800
        #     global_invocation_id, # noqa: E800
        #     [cgutils.int32_t(0), cgutils.int32_t(0)], # noqa: E800
        #     inbounds=True, # noqa: E800
        # ) # noqa: E800
        # res = builder.load(res, align=32) # noqa: E800

        res = builder.extract_element(
            builder.load(index_const),
            dim,
        )

        # Generating same check as sycl does. Did they add it to avoid pointer
        # bitcast on special constant?
        max_int32 = llvmir.Constant(res.type, 2147483648)
        cmp = builder.icmp_unsigned("<", res, max_int32)

        inst = builder.assume(cmp)
        # TODO: tail does not always work
        inst.tail = "tail"

        return res

    return sig, _intrinsic_spirv_global_index_const_gen


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_spirv_global_invocation_id(
    ty_context, ty_dim  # pylint: disable=unused-argument
):
    """Generates instruction to get index from BuiltInGlobalInvocationId."""
    return _intrinsic_spirv_global_index_const(
        ty_context, ty_dim, "BuiltInGlobalInvocationId"
    )


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_spirv_local_invocation_id(
    ty_context, ty_dim  # pylint: disable=unused-argument
):
    """Generates instruction to get index from BuiltInLocalInvocationId."""
    return _intrinsic_spirv_global_index_const(
        ty_context, ty_dim, "BuiltInLocalInvocationId"
    )


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_spirv_global_size(
    ty_context, ty_dim  # pylint: disable=unused-argument
):
    """Generates instruction to get index from BuiltInGlobalSize."""
    return _intrinsic_spirv_global_index_const(
        ty_context, ty_dim, "BuiltInGlobalSize"
    )


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_spirv_workgroup_size(
    ty_context, ty_dim  # pylint: disable=unused-argument
):
    """Generates instruction to get index from BuiltInWorkgroupSize."""
    return _intrinsic_spirv_global_index_const(
        ty_context, ty_dim, "BuiltInWorkgroupSize"
    )


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_spirv_workgroup_id(
    ty_context, ty_dim  # pylint: disable=unused-argument
):
    """Generates instruction to get index from BuiltInWorkgroupId."""
    return _intrinsic_spirv_global_index_const(
        ty_context, ty_dim, "BuiltInWorkgroupId"
    )


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_spirv_numworkgroups(
    ty_context, ty_dim  # pylint: disable=unused-argument
):
    """Generates instruction to get index from BuiltInNumWorkgroups."""
    return _intrinsic_spirv_global_index_const(
        ty_context, ty_dim, "BuiltInNumWorkgroups"
    )


def generate_index_overload(_type, _intrinsic):
    """Generates overload for the index method that generates specific IR from
    provided intrinsic."""

    def ol_item_gen_index(item, dim):
        """SPIR-V overload for :meth:`numba_dpex.kernel_api.<_type>.<method>`.

        Generates the same LLVM IR instruction as dpcpp for the
        `sycl::<type>::<method>` function.

        Raises:
            TypingError: When argument is not an integer.
        """
        if not isinstance(item, _type):
            raise TypingError(
                f"Expected an item should to be an {_type} value, but "
                f"encountered {type(item)}"
            )

        if not isinstance(dim, types.Integer):
            raise TypingError(
                f"Expected an {_type}'s dim should to be an Integer value, but "
                f"encountered {type(dim)}"
            )

        # pylint: disable=unused-argument
        def ol_item_get_index_impl(item, dim):
            # TODO: call in reverse index once index reversing is removed from
            # kernel submission
            # pylint: disable=no-value-for-parameter
            return _intrinsic(dim)

        return ol_item_get_index_impl

    return ol_item_gen_index


def register_index_const_methods():
    """Register indexing related methods that can be defined as spirv const."""
    _index_const_overload_methods = [
        (ItemType, "get_id", _intrinsic_spirv_global_invocation_id),
        (ItemType, "get_range", _intrinsic_spirv_global_size),
        (NdItemType, "get_global_id", _intrinsic_spirv_global_invocation_id),
        (NdItemType, "get_local_id", _intrinsic_spirv_local_invocation_id),
        (NdItemType, "get_global_range", _intrinsic_spirv_global_size),
        (NdItemType, "get_local_range", _intrinsic_spirv_workgroup_size),
        (GroupType, "get_group_id", _intrinsic_spirv_workgroup_id),
        (GroupType, "get_group_range", _intrinsic_spirv_numworkgroups),
        (GroupType, "get_local_range", _intrinsic_spirv_workgroup_size),
    ]

    for index_overload in _index_const_overload_methods:
        _type, method, _intrinsic = index_overload

        ol_index_func = generate_index_overload(_type, _intrinsic)

        overload_method(_type, method, target=SPIRV_TARGET_NAME)(ol_index_func)


register_index_const_methods()


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_get_group(
    ty_context, ty_nd_item: NdItemType  # pylint: disable=unused-argument
):
    """Generates group with a dimension of nd_item."""

    if not isinstance(ty_nd_item, NdItemType):
        raise TypingError(
            f"Expected an NdItemType value, but encountered {ty_nd_item}"
        )

    ty_group = GroupType(ty_nd_item.ndim)
    sig = ty_group(ty_nd_item)

    # pylint: disable=unused-argument
    def _intrinsic_get_group_gen(context, builder, sig, args):
        group_struct = cgutils.create_struct_proxy(ty_group)(context, builder)
        # pylint: disable=protected-access
        return group_struct._getvalue()

    return sig, _intrinsic_get_group_gen


@overload_method(NdItemType, "get_group", target=SPIRV_TARGET_NAME)
def ol_nd_item_get_group(nd_item):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.NdItem.get_group`.

    Generates the same LLVM IR instruction as dpcpp for the
    `sycl::nd_item::get_group` function.

    Raises:
        TypingError: When argument is not NdItem.
    """
    if not isinstance(nd_item, NdItemType):
        # since it is a method overload, this error should not be reached
        raise TypingError(
            "Expected a nd_item should to be a NdItem value, but "
            f"encountered {type(nd_item)}"
        )

    # pylint: disable=unused-argument
    def ol_nd_item_get_group_impl(nd_item):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_get_group(nd_item)

    return ol_nd_item_get_group_impl


@overload_attribute(GroupType, "dimensions", target=SPIRV_TARGET_NAME)
@overload_attribute(ItemType, "dimensions", target=SPIRV_TARGET_NAME)
@overload_attribute(NdItemType, "dimensions", target=SPIRV_TARGET_NAME)
def ol_nd_item_dimensions(item):
    """
    SPIR-V overload for :meth:`numba_dpex.kernel_api.<generic_item>.dimensions`.

    Generates the same LLVM IR instruction as dpcpp for the
    `sycl::<generic_item>::dimensions` attribute.
    """
    dimensions = item.ndim

    # pylint: disable=unused-argument
    def ol_nd_item_get_group_impl(item):
        return dimensions

    return ol_nd_item_get_group_impl


def _generate_method_overload(method):
    """Generates naive method overload with no argument, except self."""

    def ol_method(self):  # pylint: disable=unused-argument
        return method

    return ol_method


def register_jitable_method(type_, method):
    """
    Register a regular python method that can be executed by the
    python interpreter and can be compiled into a nopython
    function when referenced by other jit'ed functions.

    Same as register_jitable, but for methods with no arguments.
    """
    overloaded_method = _generate_method_overload(method)
    overload_method(type_, method.__name__, target=SPIRV_TARGET_NAME)(
        overloaded_method
    )


register_jitable_method(ItemType, Item.get_linear_id)
register_jitable_method(ItemType, Item.get_linear_range)
register_jitable_method(NdItemType, NdItem.get_global_linear_id)
register_jitable_method(NdItemType, NdItem.get_global_linear_range)
register_jitable_method(NdItemType, NdItem.get_local_linear_range)
register_jitable_method(NdItemType, NdItem.get_local_linear_id)
register_jitable_method(GroupType, Group.get_group_linear_id)
register_jitable_method(GroupType, Group.get_group_linear_range)
register_jitable_method(GroupType, Group.get_local_linear_range)
