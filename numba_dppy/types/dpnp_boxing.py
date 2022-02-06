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
""" Boxing for dpnp.ndarray type.
"""

from ctypes import pythonapi
from functools import reduce

from llvmlite.ir import IRBuilder
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.errors import NumbaNotImplementedError
from numba.core.pythonapi import NativeValue, PythonAPI, box, unbox
from numba.np import numpy_support

from .dpnp_types import dpnp_ndarray_Type

# For UsmSharedArrayType unboxing uses implemetation for types.Array
# See numba/core/pythonapi.py function _Registry.lookup():
# for cls in typeclass.__mro__:
#     func = self.functions.get(cls)
#     if func is not None:
#         return func
# It means first registered subclass of UsmSharedArrayType is used.
# In case if no unbox registered for UsmSharedArrayType then types.Array is used.

# For dpnp_ndarray_Type we can not reuse unbox for types.Array.
# It shows error from numba/core/boxing.py
# @unbox(types.Array)
# def unbox_array(typ, obj, c):
#     ...
#         # Handle error
#     with c.builder.if_then(failed, likely=False):
#         c.pyapi.err_set_string("PyExc_TypeError",
#                                "can't unbox array from PyObject into "
#                                "native value.  The object maybe of a "
#                                "different type")
#     ...
# The reason is dpnp.ndarray not inherit from numpy.ndarray.
# Also dpnp.ndarray provides additional data which should be unboxed differently.


@unbox(dpnp_ndarray_Type)
def unbox_array(typ, obj, c):
    """
    Convert a dpnp.ndarray object to a native array structure.
    dpnp.ndarray is not a subclass of NumPy array.
    """
    return _UnboxHelper(typ, obj, c)()


class _UnboxHelper:
    def __init__(self, typ, obj, c):
        self.typ = typ
        self.obj = obj
        self.builder: IRBuilder = c.builder
        self.context: BaseContext = c.context
        self.pyapi: PythonAPI = c.pyapi
        self.unboxing_errcode = None
        self.itemsize_mismatch = None
        self.error = None
        self.value = None

    def __call__(self):
        # This is necessary because unbox_buffer() does not work on some
        # dtypes, e.g. datetime64 and timedelta64.

        array_generator = self.native_array_generator()
        array_generator.generate(self.builder)

        unboxing_generator = self.unboxing_generator()
        unboxing_generator.generate(
            self.builder, self.pyapi, array_generator.nativeary, self.obj
        )
        self.unboxing_errcode = unboxing_generator.error

        # TODO: here we have minimal typechecking by the itemsize.
        #       need to do better
        # TODO check matching dtype.
        #      currently, mismatching dtype will still work and causes
        #      potential memory corruption
        self.generate_itemsize_checking(self.builder, array_generator.nativeary)

        is_error = self.generate_error_checking(self.builder)
        self.generate_error_handling(self.builder, is_error)

        value = self.generate_value(self.builder, array_generator.nativeary)
        return NativeValue(value, is_error=is_error)

    def native_array_generator(self):
        return NativeArrayGenerator(self.context, self.typ)

    def unboxing_generator(self):
        return UnboxingGenerator(self.context)

    def generate_itemsize_checking(self, builder, nativeary):
        try:
            expected_itemsize = numpy_support.as_dtype(self.typ.dtype).itemsize
        except NumbaNotImplementedError:
            # Don't check types that can't be `as_dtype()`-ed
            self.itemsize_mismatch = cgutils.false_bit
        else:
            # generate
            loaded_itemsize = nativeary.itemsize
            expected_itemsize = loaded_itemsize.type(expected_itemsize)
            # generate
            self.itemsize_mismatch = builder.icmp_unsigned(
                "!=",
                loaded_itemsize,
                expected_itemsize,
            )

    def generate_error_checking(self, builder):
        errors = [
            cgutils.is_not_null(builder, self.unboxing_errcode),
            self.itemsize_mismatch,
            cgutils.is_not_null(builder, self.pyapi.err_occurred()),
        ]
        error = reduce(builder.or_, errors)
        return error

    def generate_error_handling(self, builder, error):
        with builder.if_then(error, likely=False):
            self.pyapi.err_set_string(
                "PyExc_TypeError",
                "can't unbox dpnp.ndarray from PyObject into "
                "native value. The object maybe of a "
                "different type",
            )

    def generate_value(self, builder, nativeary):
        return builder.load(nativeary._getpointer())


# numba/core/base.py
# class BaseContext:
#     def make_array(self, typ):
#         from numba.np import arrayobj
#         return arrayobj.make_array(typ)
# numba/np/arrayobj.py
# def make_array(array_type):
#     real_array_type = array_type.as_array
#     base = cgutils.create_struct_proxy(real_array_type)
#     ndim = real_array_type.ndim
#     class ArrayStruct(base):
#         ...
#     return ArrayStruct

# returns class like this
# class _StructProxy(object):
#     def __init__(self, context, builder, value=None, ref=None):
#     def _make_refs():
# class ValueStructProxy(_StructProxy):
#     def _get_be_type(self, datamodel):
#     def _cast_member_to_value(self, index, val):
#     def _cast_member_from_value(self, index, val):
# class ValueStructProxy_dpnp.ndarray(ValueStructProxy):
#     _fe_type = ...
# class ArrayStruct(ValueStructProxy_dpnp.ndarray):
#     def _make_refs():
#     def shape():  # @property
class NativeArrayGenerator:
    def __init__(self, context: BaseContext, typ):
        self.context = context
        self.typ = typ
        self.array_struct = self.context.make_array(self.typ)
        self.nativeary = None

    def generate(self, builder: IRBuilder):
        self.nativeary = self.array_struct(self.context, builder)


# class _StructProxy(object):
#     def __init__():
#        ...
#        outer_ref, ref = self._make_refs(ref)  # ArrayStruct._make_ref()
#        ...
#        self._outer_ref = outer_ref
#
#     def _getpointer(self):
#         return self._outer_ref
class UnboxingGenerator:
    def __init__(self, context: BaseContext):
        self.context = context
        self.error = None

    def generate(self, builder: IRBuilder, pyapi: PythonAPI, nativeary, obj):
        adaptor_generator = self.adaptor_generator()
        ptr = builder.bitcast(nativeary._getpointer(), pyapi.voidptr)

        adaptor_generator.generate(pyapi, obj, ptr)

        self.error = adaptor_generator.error

    def adaptor_generator(self):
        return SyclUsmAdaptorGenerator()


class SyclUsmAdaptorGenerator:
    def __init__(self):
        self.error = None

    def generate(self, pyapi: PythonAPI, obj, ptr):
        """See PythonAPI.nrt_adapt_ndarray_from_python()"""
        fn = self.function(pyapi)
        self.error = pyapi.builder.call(fn, (obj, ptr))

    def function(self, pyapi: PythonAPI):
        import llvmlite.llvmpy.core as lc
        from llvmlite.llvmpy.core import Type

        assert pyapi.context.enable_nrt
        fnty = Type.function(Type.int(), [pyapi.pyobj, pyapi.voidptr])
        fn = pyapi._get_function(fnty, "DPPY_RT_sycl_usm_array_from_python")
        fn.args[0].add_attribute(lc.ATTR_NO_CAPTURE)
        fn.args[1].add_attribute(lc.ATTR_NO_CAPTURE)
        return fn


# Help

# numba/core/base.py
# class BaseContext(object):
#     def make_array(self, typ):
#         from numba.np import arrayobj
#         return arrayobj.make_array(typ)

# numba/np/arrayobj.py
# def make_array(array_type):
#     real_array_type = array_type.as_array
#     base = cgutils.create_struct_proxy(real_array_type)
#     class ArrayStruct(base): ...
#     return ArrayStruct


# This tells Numba how to convert from its native representation
# of a UsmArray in a njit function back to a Python UsmArray.
@box(dpnp_ndarray_Type)
def box_array(typ, val, c):
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        # Steals NRT ref
        newary = nrt_adapt_ndarray_to_python(c.pyapi, typ, val, dtypeptr)
        return newary
    else:
        parent = nativeary.parent
        c.pyapi.incref(parent)
        return parent


def nrt_adapt_ndarray_to_python(pyapi: PythonAPI, aryty, ary, dtypeptr):

    from numba.core import types

    assert pyapi.context.enable_nrt, "NRT required"

    fn = get_box_function(pyapi)

    aryptr = cgutils.alloca_once_value(pyapi.builder, ary)
    ptr = pyapi.builder.bitcast(aryptr, pyapi.voidptr)

    # Embed the Python type of the array (maybe subclass) in the LLVM IR.
    serialized = pyapi.serialize_object(aryty.box_type)
    serial_aryty_pytype = pyapi.unserialize(serialized)

    ndim = pyapi.context.get_constant(types.int32, aryty.ndim)
    writable = pyapi.context.get_constant(types.int32, int(aryty.mutable))

    args = [ptr, serial_aryty_pytype, ndim, writable, dtypeptr]
    return pyapi.builder.call(fn, args)


def get_box_function(pyapi: PythonAPI):
    from llvmlite.ir import IntType
    from llvmlite.llvmpy.core import ATTR_NO_CAPTURE, Type

    args = [pyapi.voidptr, pyapi.pyobj, IntType(32), IntType(32), pyapi.pyobj]
    fnty = Type.function(pyapi.pyobj, args)
    fn = pyapi._get_function(fnty, "DPPY_RT_sycl_usm_array_to_python_acqref")
    fn.args[0].add_attribute(ATTR_NO_CAPTURE)
    return fn
