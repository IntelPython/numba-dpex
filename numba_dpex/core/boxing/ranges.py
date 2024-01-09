# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

from numba.core import cgutils, types
from numba.extending import NativeValue, box, unbox

from numba_dpex.core.types import NdRangeType, RangeType
from numba_dpex.kernel_api import NdRange, Range


@unbox(RangeType)
def unbox_range(typ, obj, c):
    """
    Converts a Python Range object to numba-dpex's native struct representation
    for RangeType.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    range_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    with ExitStack() as stack:
        range_attr_native_value_map = {
            "ndim": None,
            "dim0": None,
            "dim1": None,
            "dim2": None,
        }

        for attr in range_attr_native_value_map.keys():
            attr_obj = c.pyapi.object_getattr_string(obj, attr)
            with cgutils.early_exit_if_null(c.builder, stack, attr_obj):
                c.builder.store(cgutils.true_bit, is_error_ptr)
            attr_native = c.unbox(types.int64, attr_obj)
            c.pyapi.decref(attr_obj)
            with cgutils.early_exit_if(c.builder, stack, attr_native.is_error):
                c.builder.store(cgutils.true_bit, is_error_ptr)
            range_attr_native_value_map[attr] = attr_native

        range_struct.ndim = range_attr_native_value_map["ndim"].value
        range_struct.dim0 = range_attr_native_value_map["dim0"].value
        range_struct.dim1 = range_attr_native_value_map["dim1"].value
        range_struct.dim2 = range_attr_native_value_map["dim2"].value

    return NativeValue(
        range_struct._getvalue(), is_error=c.builder.load(is_error_ptr)
    )


@unbox(NdRangeType)
def unbox_ndrange(typ, obj, c):
    """
    Converts a Python Range object to numba-dpex's native struct representation
    for NdRangeType.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    ndrange_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    with ExitStack() as stack:
        ndrange_attr_native_value_map = {
            "global_range": None,
            "local_range": None,
        }

        for attr in ndrange_attr_native_value_map.keys():
            attr_obj = c.pyapi.object_getattr_string(obj, attr)
            with cgutils.early_exit_if_null(c.builder, stack, attr_obj):
                c.builder.store(cgutils.true_bit, is_error_ptr)
            attr_native = c.unbox(RangeType(typ.ndim), attr_obj)
            c.pyapi.decref(attr_obj)
            with cgutils.early_exit_if(c.builder, stack, attr_native.is_error):
                c.builder.store(cgutils.true_bit, is_error_ptr)
            ndrange_attr_native_value_map[attr] = attr_native

        global_range_struct = ndrange_attr_native_value_map[
            "global_range"
        ].value
        local_range_struct = ndrange_attr_native_value_map["local_range"].value

        range_datamodel = c.context.data_model_manager.lookup(
            RangeType(typ.ndim)
        )
        ndrange_struct.ndim = c.builder.extract_value(
            global_range_struct,
            range_datamodel.get_field_position("ndim"),
        )
        ndrange_struct.gdim0 = c.builder.extract_value(
            global_range_struct,
            range_datamodel.get_field_position("dim0"),
        )
        ndrange_struct.gdim1 = c.builder.extract_value(
            global_range_struct,
            range_datamodel.get_field_position("dim1"),
        )
        ndrange_struct.gdim2 = c.builder.extract_value(
            global_range_struct,
            range_datamodel.get_field_position("dim2"),
        )
        ndrange_struct.ldim0 = c.builder.extract_value(
            local_range_struct,
            range_datamodel.get_field_position("dim0"),
        )
        ndrange_struct.ldim1 = c.builder.extract_value(
            local_range_struct,
            range_datamodel.get_field_position("dim1"),
        )
        ndrange_struct.ldim2 = c.builder.extract_value(
            local_range_struct,
            range_datamodel.get_field_position("dim2"),
        )

    return NativeValue(
        ndrange_struct._getvalue(), is_error=c.builder.load(is_error_ptr)
    )


@box(RangeType)
def box_range(typ, val, c):
    """
    Convert a native range structure to a Range object.
    """
    ret_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    fail_obj = c.pyapi.get_null_object()

    with ExitStack() as stack:
        range_struct = cgutils.create_struct_proxy(typ)(
            c.context, c.builder, value=val
        )

        dim0_obj = c.box(types.int64, range_struct.dim0)
        with cgutils.early_exit_if_null(c.builder, stack, dim0_obj):
            c.builder.store(fail_obj, ret_ptr)
        dim1_obj = c.box(types.int64, range_struct.dim1)
        with cgutils.early_exit_if_null(c.builder, stack, dim1_obj):
            c.builder.store(fail_obj, ret_ptr)
        dim2_obj = c.box(types.int64, range_struct.dim2)
        with cgutils.early_exit_if_null(c.builder, stack, dim2_obj):
            c.builder.store(fail_obj, ret_ptr)

        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Range))
        with cgutils.early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(dim0_obj)
            c.pyapi.decref(dim1_obj)
            c.pyapi.decref(dim2_obj)
            c.builder.store(fail_obj, ret_ptr)
        # NOTE: The result of this call is not checked as the clean up
        # has to occur regardless of whether it is successful. If it
        # fails `res` is set to NULL and a Python exception is set.
        if typ.ndim == 1:
            res = c.pyapi.call_function_objargs(class_obj, (dim0_obj,))
        elif typ.ndim == 2:
            res = c.pyapi.call_function_objargs(class_obj, (dim0_obj, dim1_obj))
        elif typ.ndim == 3:
            res = c.pyapi.call_function_objargs(
                class_obj, (dim0_obj, dim1_obj, dim2_obj)
            )
        else:
            raise ValueError("Cannot unbox Range instance.")

        c.pyapi.decref(dim0_obj)
        c.pyapi.decref(dim1_obj)
        c.pyapi.decref(dim2_obj)
        c.pyapi.decref(class_obj)
        c.builder.store(res, ret_ptr)

    return c.builder.load(ret_ptr)


@box(NdRangeType)
def box_ndrange(typ, val, c):
    """
    Convert a native range structure to a Range object.
    """
    ret_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    fail_obj = c.pyapi.get_null_object()

    with ExitStack() as stack:
        ndrange_struct = cgutils.create_struct_proxy(typ)(
            c.context, c.builder, value=val
        )
        grange_struct = cgutils.create_struct_proxy(RangeType(typ.ndim))(
            c.context, c.builder
        )
        lrange_struct = cgutils.create_struct_proxy(RangeType(typ.ndim))(
            c.context, c.builder
        )
        grange_struct.ndim = ndrange_struct.ndim
        grange_struct.dim0 = ndrange_struct.gdim0
        grange_struct.dim1 = ndrange_struct.gdim1
        grange_struct.dim2 = ndrange_struct.gdim2
        lrange_struct.dim0 = ndrange_struct.ldim0
        lrange_struct.dim1 = ndrange_struct.ldim1
        lrange_struct.dim2 = ndrange_struct.ldim2

        grange_obj = c.box(RangeType(typ.ndim), grange_struct._getvalue())
        with cgutils.early_exit_if_null(c.builder, stack, grange_obj):
            c.builder.store(fail_obj, ret_ptr)

        lrange_obj = c.box(RangeType(typ.ndim), lrange_struct._getvalue())
        with cgutils.early_exit_if_null(c.builder, stack, lrange_obj):
            c.builder.store(fail_obj, ret_ptr)

        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(NdRange))
        with cgutils.early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(grange_obj)
            c.pyapi.decref(lrange_obj)
            c.builder.store(fail_obj, ret_ptr)

        # NOTE: The result of this call is not checked as the clean up
        # has to occur regardless of whether it is successful. If it
        # fails `res` is set to NULL and a Python exception is set.

        res = c.pyapi.call_function_objargs(class_obj, (grange_obj, lrange_obj))

        c.pyapi.decref(grange_obj)
        c.pyapi.decref(lrange_obj)
        c.pyapi.decref(class_obj)
        c.builder.store(res, ret_ptr)

    return c.builder.load(ret_ptr)
