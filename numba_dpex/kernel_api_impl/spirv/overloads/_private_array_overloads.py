# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements the SPIR-V overloads for the kernel_api.PrivateArray class.
"""


import operator
from functools import reduce

import llvmlite.ir as llvmir
from llvmlite.ir.builder import IRBuilder
from numba.core import cgutils, types
from numba.core.typing.npydecl import parse_dtype as _ty_parse_dtype
from numba.core.typing.npydecl import parse_shape as _ty_parse_shape
from numba.core.typing.templates import Signature
from numba.extending import type_callable

from numba_dpex.core.types import USMNdArray
from numba_dpex.kernel_api import PrivateArray
from numba_dpex.kernel_api.memory_enums import AddressSpace
from numba_dpex.kernel_api_impl.spirv.arrayobj import (
    np_cfarray,
    require_literal,
)
from numba_dpex.kernel_api_impl.spirv.target import SPIRVTypingContext

from ._registry import lower


@type_callable(PrivateArray)
def type_interval(context):  # pylint: disable=unused-argument
    """Sets type of the constructor for the class
    class:`numba_dpex.kernel_api.PrivateArray`.

    Raises:
        errors.TypingError: If the shape argument is not a shape compatible
            type.
        errors.TypingError: If the dtype argument is not a dtype compatible
            type.
    """

    def typer(shape, dtype, fill_zeros=types.BooleanLiteral(False)):
        require_literal(shape)
        require_literal(fill_zeros)

        return USMNdArray(
            dtype=_ty_parse_dtype(dtype),
            ndim=_ty_parse_shape(shape),
            layout="C",
            addrspace=AddressSpace.PRIVATE.value,
        )

    return typer


@lower(PrivateArray, types.IntegerLiteral, types.Any, types.BooleanLiteral)
@lower(PrivateArray, types.Tuple, types.Any, types.BooleanLiteral)
@lower(PrivateArray, types.UniTuple, types.Any, types.BooleanLiteral)
@lower(PrivateArray, types.IntegerLiteral, types.Any)
@lower(PrivateArray, types.Tuple, types.Any)
@lower(PrivateArray, types.UniTuple, types.Any)
def dpex_private_array_lower(
    context: SPIRVTypingContext,
    builder: IRBuilder,
    sig: Signature,
    args: list[llvmir.Value],
):
    """Implements lower for the class:`numba_dpex.kernel_api.PrivateArray`"""
    shape = args[0]
    ty_shape = sig.args[0]
    if len(sig.args) == 3:
        fill_zeros = sig.args[-1].literal_value
    else:
        fill_zeros = False
    ty_array = sig.return_type

    # Allocate data on stack
    data = cgutils.alloca_once(
        builder,
        context.get_data_type(ty_array.dtype),
        size=(
            reduce(operator.mul, [s.literal_value for s in ty_shape])
            if isinstance(ty_shape, types.BaseTuple)
            else ty_shape.literal_value
        ),
    )

    ary = np_cfarray(context, builder, ty_array, ty_shape, shape, data)

    if fill_zeros:
        cgutils.memset(
            builder, ary.data, builder.mul(ary.itemsize, ary.nitems), 0
        )

    return ary._getvalue()  # pylint: disable=protected-access
