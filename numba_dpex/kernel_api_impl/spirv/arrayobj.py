# SPDX-FileCopyrightText: 2012 - 2024 Anaconda Inc.
# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: BSD-2-Clause

"""Contains SPIR-V specific array functions."""

from typing import Union

import llvmlite.ir as llvmir
from llvmlite.ir.builder import IRBuilder
from numba.core import cgutils, errors, types
from numba.core.base import BaseContext
from numba.np.arrayobj import (
    basic_indexing,
    get_itemsize,
    load_item,
    make_array,
)

from numba_dpex.core.types import USMNdArray


def populate_array(
    arraystruct, data, shape, strides, itemsize
):  # pylint: disable=too-many-arguments,too-many-locals
    """
    Helper function for populating array structures.

    The function is copied from upstream Numba and modified to support the
    USMNdArray data type that uses a different data model on SYCL devices
    than the upstream types.Array data type. USMNdArray data model does not
    have the ``parent`` and ``meminfo`` fields. This function intended to be
    used only in the SPIRVKernelTarget.

    *shape* and *strides* can be Python tuples or LLVM arrays.
    """
    context = arraystruct._context  # pylint: disable=protected-access
    builder = arraystruct._builder  # pylint: disable=protected-access
    datamodel = arraystruct._datamodel  # pylint: disable=protected-access
    # doesn't matter what this array type instance is, it's just to get the
    # fields for the data model of the standard array type in this context
    standard_array = USMNdArray(ndim=1, layout="C", dtype=types.float64)
    standard_array_type_datamodel = context.data_model_manager[standard_array]
    required_fields = set(standard_array_type_datamodel._fields)
    datamodel_fields = set(datamodel._fields)
    # Make sure that the presented array object has a data model that is
    # close enough to an array for this function to proceed.
    if (required_fields & datamodel_fields) != required_fields:
        missing = required_fields - datamodel_fields
        msg = (
            f"The datamodel for type {arraystruct} is missing "
            f"field{'s' if len(missing) > 1 else ''} {missing}."
        )
        raise ValueError(msg)

    intp_t = context.get_value_type(types.intp)
    if isinstance(shape, (tuple, list)):
        shape = cgutils.pack_array(builder, shape, intp_t)
    if isinstance(strides, (tuple, list)):
        strides = cgutils.pack_array(builder, strides, intp_t)
    if isinstance(itemsize, int):
        itemsize = intp_t(itemsize)

    attrs = {
        "shape": shape,
        "strides": strides,
        "data": data,
        "itemsize": itemsize,
    }

    # Calc num of items from shape
    nitems = context.get_constant(types.intp, 1)
    unpacked_shape = cgutils.unpack_tuple(builder, shape, shape.type.count)
    # (note empty shape => 0d array therefore nitems = 1)
    for axlen in unpacked_shape:
        nitems = builder.mul(nitems, axlen, flags=["nsw"])
    attrs["nitems"] = nitems

    # Make sure that we have all the fields
    got_fields = set(attrs.keys())
    if got_fields != required_fields:
        raise ValueError(f"missing {required_fields - got_fields}")

    # Set field value
    for k, v in attrs.items():
        setattr(arraystruct, k, v)

    return arraystruct


def make_view(
    context, builder, ary, return_type, data, shapes, strides
):  # pylint: disable=too-many-arguments
    """
    Build a view over the given array with the given parameters.

    This is analog of numpy.np.arrayobj.make_view without parent and
    meminfo fields, because they don't make sense on device. This function
    intended to be used only in kernel targets.
    """
    retary = make_array(return_type)(context, builder)
    context.populate_array(
        retary, data=data, shape=shapes, strides=strides, itemsize=ary.itemsize
    )
    return retary


def _getitem_array_generic(
    context, builder, return_type, aryty, ary, index_types, indices
):  # pylint: disable=too-many-arguments
    """
    Return the result of indexing *ary* with the given *indices*,
    returning either a scalar or a view.

    This is analog of numpy.np.arrayobj._getitem_array_generic without parent
    and meminfo fields, because they don't make sense on device. This function
    intended to be used only in kernel targets.
    """
    dataptr, view_shapes, view_strides = basic_indexing(
        context,
        builder,
        aryty,
        ary,
        index_types,
        indices,
        boundscheck=context.enable_boundscheck,
    )

    if isinstance(return_type, types.Buffer):
        # Build array view
        retary = make_view(
            context,
            builder,
            ary,
            return_type,
            dataptr,
            view_shapes,
            view_strides,
        )
        return retary._getvalue()  # pylint: disable=protected-access

    # Load scalar from 0-d result
    assert not view_shapes
    return load_item(context, builder, aryty, dataptr)


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


def np_cfarray(  # pylint: disable=too-many-arguments
    context: BaseContext,
    builder: IRBuilder,
    ty_array: types.Array,
    ty_shape: Union[types.IntegerLiteral, types.BaseTuple],
    shape: llvmir.Value,
    data: llvmir.Value,
):
    """Makes numpy-like array and fills it's data depending on the context's
    datamodel.

    Generic version of numba.np.arrayobj.np_cfarray so that it can be used
    not only as intrinsic, but inside instruction generation.

    TODO: upstream changes to numba.
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
