# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements the SPIR-V overloads for the kernel_api.PrivateArray class.
"""


import llvmlite.ir as llvmir
from llvmlite.ir.builder import IRBuilder
from numba.core import cgutils
from numba.core.typing.npydecl import parse_dtype as _ty_parse_dtype
from numba.core.typing.npydecl import parse_shape as _ty_parse_shape
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, overload

from numba_dpex.core.types import USMNdArray
from numba_dpex.experimental.target import DpexExpKernelTypingContext
from numba_dpex.kernel_api import PrivateArray
from numba_dpex.kernel_api_impl.spirv.arrayobj import (
    make_spirv_generic_array_on_stack,
    require_literal,
)
from numba_dpex.utils import address_space as AddressSpace

from ..target import DPEX_KERNEL_EXP_TARGET_NAME


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_private_array_ctor(
    ty_context,  # pylint: disable=unused-argument
    ty_shape,
    ty_dtype,
    ty_fill_zeros,
):
    require_literal(ty_shape)
    require_literal(ty_fill_zeros)

    ty_array = USMNdArray(
        dtype=_ty_parse_dtype(ty_dtype),
        ndim=_ty_parse_shape(ty_shape),
        layout="C",
        addrspace=AddressSpace.PRIVATE,
    )

    sig = ty_array(ty_shape, ty_dtype, ty_fill_zeros)

    def codegen(
        context: DpexExpKernelTypingContext,
        builder: IRBuilder,
        sig: Signature,
        args: list[llvmir.Value],
    ):
        shape = args[0]
        ty_shape = sig.args[0]
        ty_fill_zeros = sig.args[-1]
        ty_array = sig.return_type

        ary = make_spirv_generic_array_on_stack(
            context, builder, ty_array, ty_shape, shape
        )

        if ty_fill_zeros.literal_value:
            cgutils.memset(
                builder, ary.data, builder.mul(ary.itemsize, ary.nitems), 0
            )

        return ary._getvalue()  # pylint: disable=protected-access

    return (
        sig,
        codegen,
    )


@overload(
    PrivateArray,
    prefer_literal=True,
    target=DPEX_KERNEL_EXP_TARGET_NAME,
)
def ol_private_array_ctor(
    shape,
    dtype,
    fill_zeros=False,
):
    """Overload of the constructor for the class
    class:`numba_dpex.kernel_api.PrivateArray`.

    Raises:
        errors.TypingError: If the shape argument is not a shape compatible
            type.
        errors.TypingError: If the dtype argument is not a dtype compatible
            type.
    """

    def ol_private_array_ctor_impl(
        shape,
        dtype,
        fill_zeros=False,
    ):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_private_array_ctor(shape, dtype, fill_zeros)

    return ol_private_array_ctor_impl
