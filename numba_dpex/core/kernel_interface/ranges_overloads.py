# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from numba.core import cgutils, errors, types
from numba.extending import intrinsic, overload

from numba_dpex.kernel_api import NdRange, Range

# can't import name because of the circular import
DPEX_TARGET_NAME = "dpex"


@intrinsic(target=DPEX_TARGET_NAME)
def _intrin_range_alloc(typingctx, ty_dim0, ty_dim1, ty_dim2, ty_range):
    ty_retty = ty_range.instance_type
    sig = ty_retty(
        ty_dim0,
        ty_dim1,
        ty_dim2,
        ty_range,
    )

    def codegen(context, builder, sig, args):
        typ = sig.return_type
        dim0, dim1, dim2, _ = args
        range_struct = cgutils.create_struct_proxy(typ)(context, builder)
        range_struct.dim0 = dim0

        if not isinstance(sig.args[1], types.NoneType):
            range_struct.dim1 = dim1
        else:
            range_struct.dim1 = llvmir.Constant(
                llvmir.types.IntType(64), Range._UNDEFINED_DIMENSION
            )

        if not isinstance(sig.args[2], types.NoneType):
            range_struct.dim2 = dim2
        else:
            range_struct.dim2 = llvmir.Constant(
                llvmir.types.IntType(64), Range._UNDEFINED_DIMENSION
            )

        range_struct.ndim = llvmir.Constant(llvmir.types.IntType(64), typ.ndim)

        return range_struct._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def _intrin_ndrange_alloc(
    typingctx, ty_global_range, ty_local_range, ty_ndrange
):
    ty_retty = ty_ndrange.instance_type
    sig = ty_retty(
        ty_global_range,
        ty_local_range,
        ty_ndrange,
    )

    def codegen(context, builder, sig, args):
        typ = sig.return_type

        range_datamodel = context.data_model_manager.lookup(ty_global_range)

        global_range, local_range, _ = args
        ndrange_struct = cgutils.create_struct_proxy(typ)(context, builder)
        ndrange_struct.ndim = llvmir.Constant(
            llvmir.types.IntType(64), typ.ndim
        )
        ndrange_struct.gdim0 = builder.extract_value(
            global_range,
            range_datamodel.get_field_position("dim0"),
        )
        ndrange_struct.gdim1 = builder.extract_value(
            global_range,
            range_datamodel.get_field_position("dim1"),
        )
        ndrange_struct.gdim2 = builder.extract_value(
            global_range,
            range_datamodel.get_field_position("dim2"),
        )
        ndrange_struct.ldim0 = builder.extract_value(
            local_range,
            range_datamodel.get_field_position("dim0"),
        )
        ndrange_struct.ldim1 = builder.extract_value(
            local_range,
            range_datamodel.get_field_position("dim1"),
        )
        ndrange_struct.ldim2 = builder.extract_value(
            local_range,
            range_datamodel.get_field_position("dim2"),
        )

        return ndrange_struct._getvalue()

    return sig, codegen


@overload(Range, target=DPEX_TARGET_NAME)
def _ol_range_init(dim0, dim1=None, dim2=None):
    """Numba overload of the Range constructor to make it usable inside an
    njit and dpjit decorated function.

    """
    from numba_dpex.core.types import RangeType

    ndims = 1
    ty_optional_dims = (dim1, dim2)

    # A Range should at least have the 0th dimension populated
    if not isinstance(dim0, types.Integer):
        raise errors.TypingError(
            "Expected a Range's dimension should to be an Integer value, but "
            "encountered " + dim0.name
        )

    for ty_dim in ty_optional_dims:
        if isinstance(ty_dim, types.Integer):
            ndims += 1
        elif ty_dim is not None:
            raise errors.TypingError(
                "Expected a Range's dimension to be an Integer value, "
                f"but {type(ty_dim)} was provided."
            )

    ret_ty = RangeType(ndims)

    def impl(dim0, dim1=None, dim2=None):
        return _intrin_range_alloc(dim0, dim1, dim2, ret_ty)

    return impl


@overload(NdRange, target=DPEX_TARGET_NAME)
def _ol_ndrange_init(global_range, local_range):
    """Numba overload of the NdRange constructor to make it usable inside an
    njit and dpjit decorated function.

    """
    from numba_dpex.core.exceptions import UnmatchedNumberOfRangeDimsError
    from numba_dpex.core.types import NdRangeType, RangeType

    if not isinstance(global_range, RangeType):
        raise errors.TypingError(
            "Only global range values specified as a Range are "
            "supported inside dpjit"
        )

    if not isinstance(local_range, RangeType):
        raise errors.TypingError(
            "Only local range values specified as a Range are "
            "supported inside dpjit"
        )

    if not global_range.ndim == local_range.ndim:
        raise UnmatchedNumberOfRangeDimsError(
            kernel_name="",
            global_ndims=global_range.ndim,
            local_ndims=local_range.ndim,
        )

    ret_ty = NdRangeType(global_range.ndim)

    def impl(global_range, local_range):
        return _intrin_ndrange_alloc(global_range, local_range, ret_ty)

    return impl
