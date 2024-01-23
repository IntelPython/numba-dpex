# SPDX-FileCopyrightText: 2012 - 2024 Anaconda Inc.
# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-2-Clause

"""
This package contains implementation of some numpy.np.arrayobj functions without
parent and meminfo fields required, because they don't make sense on device.
These functions intended to be used only in kernel targets like local/private or
usm array view.
"""


from numba.core import cgutils, types
from numba.np import arrayobj as np_arrayobj

from numba_dpex.core import types as dpex_types


def populate_array(array, data, shape, strides, itemsize):
    """
    Helper function for populating array structures.
    This avoids forgetting to set fields.

    *shape* and *strides* can be Python tuples or LLVM arrays.

    This is analog of numpy.np.arrayobj.populate_array without parent and
    meminfo fields, because they don't make sense on device. This function
    intended to be used only in kernel targets.
    """
    context = array._context
    builder = array._builder
    datamodel = array._datamodel
    # doesn't matter what this array type instance is, it's just to get the
    # fields for the datamodel of the standard array type in this context
    standard_array = dpex_types.Array(types.float64, 1, "C")
    standard_array_type_datamodel = context.data_model_manager[standard_array]
    required_fields = set(standard_array_type_datamodel._fields)
    datamodel_fields = set(datamodel._fields)
    # Make sure that the presented array object has a data model that is close
    # enough to an array for this function to proceed.
    if (required_fields & datamodel_fields) != required_fields:
        missing = required_fields - datamodel_fields
        msg = (
            f"The datamodel for type {array._fe_type} is missing "
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

    attrs = dict(
        shape=shape,
        strides=strides,
        data=data,
        itemsize=itemsize,
    )

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
        raise ValueError("missing {0}".format(required_fields - got_fields))

    # Set field value
    for k, v in attrs.items():
        setattr(array, k, v)

    return array


def make_view(context, builder, aryty, ary, return_type, data, shapes, strides):
    """
    Build a view over the given array with the given parameters.

    This is analog of numpy.np.arrayobj.make_view without parent and
    meminfo fields, because they don't make sense on device. This function
    intended to be used only in kernel targets.
    """
    retary = np_arrayobj.make_array(return_type)(context, builder)
    populate_array(
        retary, data=data, shape=shapes, strides=strides, itemsize=ary.itemsize
    )
    return retary


def _getitem_array_generic(
    context, builder, return_type, aryty, ary, index_types, indices
):
    """
    Return the result of indexing *ary* with the given *indices*,
    returning either a scalar or a view.

    This is analog of numpy.np.arrayobj._getitem_array_generic without parent
    and meminfo fields, because they don't make sense on device. This function
    intended to be used only in kernel targets.
    """
    dataptr, view_shapes, view_strides = np_arrayobj.basic_indexing(
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
            aryty,
            ary,
            return_type,
            dataptr,
            view_shapes,
            view_strides,
        )
        return retary._getvalue()
    else:
        # Load scalar from 0-d result
        assert not view_shapes
        return np_arrayobj.load_item(context, builder, aryty, dataptr)
