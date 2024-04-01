# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Contains SPIR-V specific array functions."""

import operator
from functools import reduce
from typing import Union

import llvmlite.ir as llvmir
from llvmlite.ir.builder import IRBuilder
from numba.core import cgutils, errors, types
from numba.core.base import BaseContext
from numba.np.arrayobj import get_itemsize


def require_literal(literal_type: types.Type):
    """Checks if the numba type is Literal. If iterable object is passed,
    checks that every element is Literal.

    Raises:
        TypingError: When argument is not Iterable.
    """
    if not hasattr(literal_type, "__len__"):
        if not isinstance(literal_type, types.Literal):
            raise errors.TypingError("requires literal type")
        return

    for i, _ in enumerate(literal_type):
        if not isinstance(literal_type[i], types.Literal):
            raise errors.TypingError(
                "requires each element of tuple literal type"
            )


def make_spirv_array(  # pylint: disable=too-many-arguments
    context: BaseContext,
    builder: IRBuilder,
    ty_array: types.Array,
    ty_shape: Union[types.IntegerLiteral, types.BaseTuple],
    shape: llvmir.Value,
    data: llvmir.Value,
):
    """Makes SPIR-V array and fills it data.

    Generic version of numba.np.arrayobj.np_cfarray so that it can be used
    not only as intrinsic, but inside instruction generation.
    """
    # Create array object
    ary = context.make_array(ty_array)(context, builder)

    itemsize = get_itemsize(context, ty_array)
    ll_itemsize = cgutils.intp_t(itemsize)

    if isinstance(ty_shape, types.BaseTuple):
        shapes = cgutils.unpack_tuple(builder, shape)
    else:
        ty_shape = (ty_shape,)
        shapes = (shape,)
    shapes = [
        context.cast(builder, value, fromty, types.intp)
        for fromty, value in zip(ty_shape, shapes)
    ]

    off = ll_itemsize
    strides = []
    if ty_array.layout == "F":
        for s in shapes:
            strides.append(off)
            off = builder.mul(off, s)
    else:
        for s in reversed(shapes):
            strides.append(off)
            off = builder.mul(off, s)
        strides.reverse()

    context.populate_array(
        ary,
        data=data,
        shape=shapes,
        strides=strides,
        itemsize=ll_itemsize,
    )

    return ary


def allocate_array_data_on_stack(
    context: BaseContext,
    builder: IRBuilder,
    ty_array: types.Array,
    ty_shape: Union[types.IntegerLiteral, types.BaseTuple],
):
    """Allocates flat array of given shape on the stack."""
    if not isinstance(ty_shape, types.BaseTuple):
        ty_shape = (ty_shape,)

    return cgutils.alloca_once(
        builder,
        context.get_data_type(ty_array.dtype),
        size=reduce(operator.mul, [s.literal_value for s in ty_shape]),
    )


def make_spirv_generic_array_on_stack(
    context: BaseContext,
    builder: IRBuilder,
    ty_array: types.Array,
    ty_shape: Union[types.IntegerLiteral, types.BaseTuple],
    shape: llvmir.Value,
):
    """Makes SPIR-V array of given shape with empty data."""
    data = allocate_array_data_on_stack(context, builder, ty_array, ty_shape)
    return make_spirv_array(context, builder, ty_array, ty_shape, shape, data)
