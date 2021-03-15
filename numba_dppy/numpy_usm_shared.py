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

import numpy as np
from inspect import getmembers, isfunction, isclass, isbuiltin
from numbers import Number
import numba
from types import FunctionType as ftype, BuiltinFunctionType as bftype
from numba import types
from numba.extending import typeof_impl, register_model, type_callable, lower_builtin
from numba.core.datamodel.registry import register_default as register_model_default
from numba.np import numpy_support
from numba.core.pythonapi import box, allocator
from llvmlite import ir
import llvmlite.llvmpy.core as lc
import llvmlite.binding as llb
from numba.core import types, cgutils, config
import builtins
import sys
from ctypes.util import find_library
from numba.core.typing.templates import builtin_registry as templates_registry
from numba.core.typing.npydecl import registry as typing_registry
from numba.core.imputils import builtin_registry as lower_registry
import importlib
import functools
import inspect
from numba.core.typing.templates import (
    CallableTemplate,
    AttributeTemplate,
    signature,
    bound_function,
)
from numba.core.typing.arraydecl import normalize_shape
from numba.np.arrayobj import _array_copy

import dpctl.dptensor.numpy_usm_shared as nus
from dpctl.dptensor.numpy_usm_shared import ndarray, functions_list, class_list
from . import target as dppy_target
from numba_dppy.dppy_array_type import DPPYArray, DPPYArrayModel


debug = config.DEBUG


def dprint(*args):
    if debug:
        print(*args)
        sys.stdout.flush()


# # This code makes it so that Numba can contain calls into the DPPLSyclInterface library.
# sycl_mem_lib = find_library('DPCTLSyclInterface')
# dprint("sycl_mem_lib:", sycl_mem_lib)
# # Load the symbols from the DPPL Sycl library.
# llb.load_library_permanently(sycl_mem_lib)

import dpctl
from dpctl.memory import MemoryUSMShared
import numba_dppy._dppy_rt

# Register the helper function in dppl_rt so that we can insert calls to them via llvmlite.
for py_name, c_address in numba_dppy._dppy_rt.c_helpers.items():
    llb.add_symbol(py_name, c_address)


class UsmSharedArrayType(DPPYArray):
    """Creates a Numba type for Numpy arrays that are stored in USM shared
    memory.  We inherit from Numba's existing Numpy array type but overload
    how this type is printed during dumping of typing information and we
    implement the special __array_ufunc__ function to determine who this
    type gets combined with scalars and regular Numpy types.
    We re-use Numpy functions as well but those are going to return Numpy
    arrays allocated in USM and we use the overloaded copy function to
    convert such USM-backed Numpy arrays into typed USM arrays."""

    def __init__(
        self,
        dtype,
        ndim,
        layout,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
    ):
        # This name defines how this type will be shown in Numba's type dumps.
        name = "UsmArray:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
        super(UsmSharedArrayType, self).__init__(
            dtype,
            ndim,
            layout,
            py_type=ndarray,
            readonly=readonly,
            name=name,
            addrspace=addrspace,
        )

    def copy(self, *args, **kwargs):
        retty = super(UsmSharedArrayType, self).copy(*args, **kwargs)
        if isinstance(retty, types.Array):
            return UsmSharedArrayType(
                dtype=retty.dtype, ndim=retty.ndim, layout=retty.layout
            )
        else:
            return retty

    # Tell Numba typing how to combine UsmSharedArrayType with other ndarray types.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            for inp in inputs:
                if not isinstance(inp, (UsmSharedArrayType, types.Array, types.Number)):
                    return None

            return UsmSharedArrayType
        else:
            return None


# This tells Numba how to create a UsmSharedArrayType when a usmarray is passed
# into a njit function.
@typeof_impl.register(ndarray)
def typeof_ta_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return UsmSharedArrayType(dtype, val.ndim, layout, readonly=readonly)


# This tells Numba to use the default Numpy ndarray data layout for
# object of type UsmArray.
register_model(UsmSharedArrayType)(DPPYArrayModel)
dppy_target.spirv_data_model_manager.register(UsmSharedArrayType, DPPYArrayModel)

# This tells Numba how to convert from its native representation
# of a UsmArray in a njit function back to a Python UsmArray.
@box(UsmSharedArrayType)
def box_array(typ, val, c):
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        # Steals NRT ref
        newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
        return newary
    else:
        parent = nativeary.parent
        c.pyapi.incref(parent)
        return parent


# This tells Numba to use this function when it needs to allocate a
# UsmArray in a njit function.
@allocator(UsmSharedArrayType)
def allocator_UsmArray(context, builder, size, align):
    context.nrt._require_nrt()

    mod = builder.module
    u32 = ir.IntType(32)

    # Get the Numba external allocator for USM memory.
    ext_allocator_fnty = ir.FunctionType(cgutils.voidptr_t, [])
    ext_allocator_fn = mod.get_or_insert_function(
        ext_allocator_fnty, name="usmarray_get_ext_allocator"
    )
    ext_allocator = builder.call(ext_allocator_fn, [])
    # Get the Numba function to allocate an aligned array with an external allocator.
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t, u32, cgutils.voidptr_t])
    fn = mod.get_or_insert_function(
        fnty, name="NRT_MemInfo_alloc_safe_aligned_external"
    )
    fn.return_value.add_attribute("noalias")
    if isinstance(align, builtins.int):
        align = context.get_constant(types.uint32, align)
    else:
        assert align.type == u32, "align must be a uint32"
    return builder.call(fn, [size, align, ext_allocator])


_registered = False


def is_usm_callback(obj):
    dprint("is_usm_callback:", obj, type(obj))
    if isinstance(obj, numba.core.runtime._nrt_python._MemInfo):
        mobj = obj
        while isinstance(mobj, numba.core.runtime._nrt_python._MemInfo):
            ea = mobj.external_allocator
            dppl_rt_allocator = numba_dppy._dppy_rt.get_external_allocator()
            dprint("Checking MemInfo:", ea)
            if ea == dppl_rt_allocator:
                return True
            mobj = mobj.parent
            if isinstance(mobj, ndarray):
                mobj = mobj.base
    return False


def numba_register():
    global _registered
    if not _registered:
        _registered = True
        ndarray.add_external_usm_checker(is_usm_callback)
        numba_register_typing()
        numba_register_lower_builtin()


# Copy a function registered as a lowerer in Numba but change the
# "np" import in Numba to point to usmarray instead of NumPy.
def copy_func_for_usmarray(f, usmarray_mod):
    import copy as cc

    # Make a copy so our change below doesn't affect anything else.
    gglobals = cc.copy(f.__globals__)
    # Make the "np"'s in the code use usmarray instead of Numba's default NumPy.
    gglobals["np"] = usmarray_mod
    # Create a new function using the original code but the new globals.
    g = ftype(f.__code__, gglobals, None, f.__defaults__, f.__closure__)
    # Some other tricks to make sure the function copy works.
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def types_replace_array(x):
    return tuple([z if z != types.Array else UsmSharedArrayType for z in x])


def numba_register_lower_builtin():
    todo = []
    todo_builtin = []
    todo_getattr = []
    todo_array_member_func = []

    # For all Numpy identifiers that have been registered for typing in Numba...
    # this registry contains functions, getattrs, setattrs, casts and constants...
    for ig in lower_registry.functions:
        impl, func, types = ig
        dprint("Numpy lowered registry functions:", impl, func, type(func), types)
        # If it is a Numpy function...
        if isinstance(func, ftype):
            dprint("is ftype")
            if func.__module__ == np.__name__:
                dprint("is Numpy module")
                # If we have overloaded that function in the usmarray module (always True right now)...
                if func.__name__ in functions_list:
                    todo.append(ig)
        if isinstance(func, bftype):
            dprint("is bftype")
            if func.__module__ == np.__name__:
                dprint("is Numpy module")
                # If we have overloaded that function in the usmarray module (always True right now)...
                if func.__name__ in functions_list:
                    todo.append(ig)
        if isinstance(func, str) and func.startswith("array."):
            todo_array_member_func.append(ig)

    for lg in lower_registry.getattrs:
        func, attr, types = lg
        dprint("Numpy lowered registry getattrs:", func, attr, types)
        types_with_usmarray = types_replace_array(types)
        if UsmSharedArrayType in types_with_usmarray:
            dprint(
                "lower_getattr:", func, type(func), attr, type(attr), types, type(types)
            )
            todo_getattr.append((func, attr, types_with_usmarray))

    for lg in todo_getattr:
        lower_registry.getattrs.append(lg)

    for impl, func, types in todo + todo_builtin:
        try:
            usmarray_func = eval("dpctl.dptensor.numpy_usm_shared." + func.__name__)
        except:
            dprint("failed to eval", func.__name__)
            continue
        dprint(
            "need to re-register lowerer for usmarray", impl, func, types, usmarray_func
        )
        new_impl = copy_func_for_usmarray(impl, nus)
        lower_registry.functions.append((new_impl, usmarray_func, types))

    for impl, func, types in todo_array_member_func:
        types_with_usmarray = types_replace_array(types)
        usmarray_func = "usm" + func
        dprint("Registering lowerer for", impl, usmarray_func, types_with_usmarray)
        new_impl = copy_func_for_usmarray(impl, nus)
        lower_registry.functions.append((new_impl, usmarray_func, types_with_usmarray))


def argspec_to_string(argspec):
    first_default_arg = len(argspec.args) - len(argspec.defaults)
    non_def = argspec.args[:first_default_arg]
    arg_zip = list(zip(argspec.args[first_default_arg:], argspec.defaults))
    combined = [a + "=" + str(b) for a, b in arg_zip]
    return ",".join(non_def + combined)


def numba_register_typing():
    todo = []
    todo_classes = []
    todo_getattr = []

    # For all Numpy identifiers that have been registered for typing in Numba...
    for ig in typing_registry.globals:
        val, typ = ig
        dprint("Numpy registered:", val, type(val), typ, type(typ))
        # If it is a Numpy function...
        if isinstance(val, (ftype, bftype)):
            # If we have overloaded that function in the usmarray module (always True right now)...
            if val.__name__ in functions_list:
                todo.append(ig)
        if isinstance(val, type):
            if isinstance(typ, numba.core.types.functions.Function):
                todo.append(ig)
            elif isinstance(typ, numba.core.types.functions.NumberClass):
                pass

    for tgetattr in templates_registry.attributes:
        dprint("Numpy getattr:", tgetattr, type(tgetattr), tgetattr.key)
        if tgetattr.key == types.Array:
            todo_getattr.append(tgetattr)

    for val, typ in todo_classes:
        dprint("todo_classes:", val, typ, type(typ))

        try:
            dptype = eval("dpctl.dptensor.numpy_usm_shared." + val.__name__)
        except:
            dprint("failed to eval", val.__name__)
            continue

        typing_registry.register_global(
            dptype, numba.core.types.NumberClass(typ.instance_type)
        )

    for val, typ in todo:
        assert len(typ.templates) == 1
        # template is the typing class to invoke generic() upon.
        template = typ.templates[0]
        dprint("need to re-register for usmarray", val, typ, typ.typing_key)
        try:
            dpval = eval("dpctl.dptensor.numpy_usm_shared." + val.__name__)
        except:
            dprint("failed to eval", val.__name__)
            continue
        dprint("--------------------------------------------------------------")
        dprint("need to re-register for usmarray", val, typ, typ.typing_key)
        dprint("val:", val, type(val), "dir val", dir(val))
        dprint("typ:", typ, type(typ), "dir typ", dir(typ))
        dprint("typing key:", typ.typing_key)
        dprint("name:", typ.name)
        dprint("key:", typ.key)
        dprint("templates:", typ.templates)
        dprint("template:", template, type(template))
        dprint("dpval:", dpval, type(dpval))
        dprint("--------------------------------------------------------------")

        class_name = "DparrayTemplate_" + val.__name__

        @classmethod
        def set_key_original(cls, key, original):
            cls.key = key
            cls.original = original

        def generic_impl(self):
            original_typer = self.__class__.original.generic(self.__class__.original)
            ot_argspec = inspect.getfullargspec(original_typer)
            astr = argspec_to_string(ot_argspec)

            typer_func = """def typer({}):
                                original_res = original_typer({})
                                if isinstance(original_res, types.Array):
                                    return UsmSharedArrayType(
                                        dtype=original_res.dtype,
                                        ndim=original_res.ndim,
                                        layout=original_res.layout
                                    )
                                return original_res""".format(
                astr, ",".join(ot_argspec.args)
            )

            try:
                gs = globals()
                ls = locals()
                gs["original_typer"] = ls["original_typer"]
                exec(typer_func, globals(), locals())
            except NameError as ne:
                print("NameError in exec:", ne)
                sys.exit(0)
            except:
                print("exec failed!", sys.exc_info()[0])
                sys.exit(0)

            try:
                exec_res = eval("typer")
            except NameError as ne:
                print("NameError in eval:", ne)
                sys.exit(0)
            except:
                print("eval failed!", sys.exc_info()[0])
                sys.exit(0)

            return exec_res

        new_usmarray_template = type(
            class_name,
            (template,),
            {"set_class_vars": set_key_original, "generic": generic_impl},
        )

        new_usmarray_template.set_class_vars(dpval, template)

        assert callable(dpval)
        type_handler = types.Function(new_usmarray_template)
        typing_registry.register_global(dpval, type_handler)

    # Handle usmarray attribute typing.
    # This explicit register_attr of a copied/modified UsmArrayAttribute
    # may be removed in the future in favor of the below commented out code
    # once we get this registration code to run after everything is registered
    # in Numba.  Right now, the attribute registrations we need are happening
    # after the registration callback that gets us here so we would miss the
    # attribute registrations we need.
    typing_registry.register_attr(UsmArrayAttribute)


class UsmArrayAttribute(AttributeTemplate):
    key = UsmSharedArrayType

    def resolve_dtype(self, ary):
        return types.DType(ary.dtype)

    def resolve_itemsize(self, ary):
        return types.intp

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_strides(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_ndim(self, ary):
        return types.intp

    def resolve_size(self, ary):
        return types.intp

    def resolve_flat(self, ary):
        return types.NumpyFlatType(ary)

    def resolve_ctypes(self, ary):
        return types.ArrayCTypes(ary)

    def resolve_flags(self, ary):
        return types.ArrayFlags(ary)

    def convert_array_to_usmarray(self, retty):
        if isinstance(retty, types.Array):
            return UsmSharedArrayType(
                dtype=retty.dtype, ndim=retty.ndim, layout=retty.layout
            )
        else:
            return retty

    def resolve_T(self, ary):
        if ary.ndim <= 1:
            retty = ary
        else:
            layout = {"C": "F", "F": "C"}.get(ary.layout, "A")
            retty = ary.copy(layout=layout)
        return retty

    def resolve_real(self, ary):
        return self._resolve_real_imag(ary, attr="real")

    def resolve_imag(self, ary):
        return self._resolve_real_imag(ary, attr="imag")

    def _resolve_real_imag(self, ary, attr):
        if ary.dtype in types.complex_domain:
            return ary.copy(dtype=ary.dtype.underlying_float, layout="A")
        elif ary.dtype in types.number_domain:
            res = ary.copy(dtype=ary.dtype)
            if attr == "imag":
                res = res.copy(readonly=True)
            return res
        else:
            msg = "cannot access .{} of array of {}"
            raise TypingError(msg.format(attr, ary.dtype))

    @bound_function("usmarray.copy")
    def resolve_copy(self, ary, args, kws):
        assert not args
        assert not kws
        retty = ary.copy(layout="C", readonly=False)
        return signature(retty)

    @bound_function("usmarray.transpose")
    def resolve_transpose(self, ary, args, kws):
        def sentry_shape_scalar(ty):
            if ty in types.number_domain:
                # Guard against non integer type
                if not isinstance(ty, types.Integer):
                    raise TypeError("transpose() arg cannot be {0}".format(ty))
                return True
            else:
                return False

        assert not kws
        if len(args) == 0:
            return signature(self.resolve_T(ary))

        if len(args) == 1:
            (shape,) = args

            if sentry_shape_scalar(shape):
                assert ary.ndim == 1
                return signature(ary, *args)

            if isinstance(shape, types.NoneType):
                return signature(self.resolve_T(ary))

            shape = normalize_shape(shape)
            if shape is None:
                return

            assert ary.ndim == shape.count
            return signature(self.resolve_T(ary).copy(layout="A"), shape)

        else:
            if any(not sentry_shape_scalar(a) for a in args):
                raise TypeError(
                    "transpose({0}) is not supported".format(", ".join(args))
                )
            assert ary.ndim == len(args)
            return signature(self.resolve_T(ary).copy(layout="A"), *args)

    @bound_function("usmarray.item")
    def resolve_item(self, ary, args, kws):
        assert not kws
        # We don't support explicit arguments as that's exactly equivalent
        # to regular indexing.  The no-argument form is interesting to
        # allow some degree of genericity when writing functions.
        if not args:
            return signature(ary.dtype)

    @bound_function("usmarray.itemset")
    def resolve_itemset(self, ary, args, kws):
        assert not kws
        # We don't support explicit arguments as that's exactly equivalent
        # to regular indexing.  The no-argument form is interesting to
        # allow some degree of genericity when writing functions.
        if len(args) == 1:
            return signature(types.none, ary.dtype)

    @bound_function("usmarray.nonzero")
    def resolve_nonzero(self, ary, args, kws):
        assert not args
        assert not kws
        # 0-dim arrays return one result array
        ndim = max(ary.ndim, 1)
        retty = types.UniTuple(UsmSharedArrayType(types.intp, 1, "C"), ndim)
        return signature(retty)

    @bound_function("usmarray.reshape")
    def resolve_reshape(self, ary, args, kws):
        def sentry_shape_scalar(ty):
            if ty in types.number_domain:
                # Guard against non integer type
                if not isinstance(ty, types.Integer):
                    raise TypeError("reshape() arg cannot be {0}".format(ty))
                return True
            else:
                return False

        assert not kws
        if ary.layout not in "CF":
            # only work for contiguous array
            raise TypeError("reshape() supports contiguous array only")

        if len(args) == 1:
            # single arg
            (shape,) = args

            if sentry_shape_scalar(shape):
                ndim = 1
            else:
                shape = normalize_shape(shape)
                if shape is None:
                    return
                ndim = shape.count
            retty = ary.copy(ndim=ndim)
            return signature(retty, shape)

        elif len(args) == 0:
            # no arg
            raise TypeError("reshape() take at least one arg")

        else:
            # vararg case
            if any(not sentry_shape_scalar(a) for a in args):
                raise TypeError(
                    "reshape({0}) is not supported".format(", ".join(map(str, args)))
                )

            retty = ary.copy(ndim=len(args))
            return signature(retty, *args)

    @bound_function("usmarray.sort")
    def resolve_sort(self, ary, args, kws):
        assert not args
        assert not kws
        if ary.ndim == 1:
            return signature(types.none)

    @bound_function("usmarray.argsort")
    def resolve_argsort(self, ary, args, kws):
        assert not args
        kwargs = dict(kws)
        kind = kwargs.pop("kind", types.StringLiteral("quicksort"))
        if not isinstance(kind, types.StringLiteral):
            raise errors.TypingError('"kind" must be a string literal')
        if kwargs:
            msg = "Unsupported keywords: {!r}"
            raise TypingError(msg.format([k for k in kwargs.keys()]))
        if ary.ndim == 1:

            def argsort_stub(kind="quicksort"):
                pass

            pysig = utils.pysignature(argsort_stub)
            sig = signature(UsmSharedArrayType(types.intp, 1, "C"), kind).replace(
                pysig=pysig
            )
            return sig

    @bound_function("usmarray.view")
    def resolve_view(self, ary, args, kws):
        from .npydecl import parse_dtype

        assert not kws
        (dtype,) = args
        dtype = parse_dtype(dtype)
        if dtype is None:
            return
        retty = ary.copy(dtype=dtype)
        return signature(retty, *args)

    @bound_function("usmarray.astype")
    def resolve_astype(self, ary, args, kws):
        from .npydecl import parse_dtype

        assert not kws
        (dtype,) = args
        dtype = parse_dtype(dtype)
        if dtype is None:
            return
        if not self.context.can_convert(ary.dtype, dtype):
            raise TypeError(
                "astype(%s) not supported on %s: "
                "cannot convert from %s to %s" % (dtype, ary, ary.dtype, dtype)
            )
        layout = ary.layout if ary.layout in "CF" else "C"
        # reset the write bit irrespective of whether the cast type is the same
        # as the current dtype, this replicates numpy
        retty = ary.copy(dtype=dtype, layout=layout, readonly=False)
        return signature(retty, *args)

    @bound_function("usmarray.ravel")
    def resolve_ravel(self, ary, args, kws):
        # Only support no argument version (default order='C')
        assert not kws
        assert not args
        return signature(ary.copy(ndim=1, layout="C"))

    @bound_function("usmarray.flatten")
    def resolve_flatten(self, ary, args, kws):
        # Only support no argument version (default order='C')
        assert not kws
        assert not args
        return signature(ary.copy(ndim=1, layout="C"))

    @bound_function("usmarray.take")
    def resolve_take(self, ary, args, kws):
        assert not kws
        (argty,) = args
        if isinstance(argty, types.Integer):
            sig = signature(ary.dtype, *args)
        elif isinstance(argty, UsmSharedArrayType):
            sig = signature(argty.copy(layout="C", dtype=ary.dtype), *args)
        elif isinstance(argty, types.List):  # 1d lists only
            sig = signature(UsmSharedArrayType(ary.dtype, 1, "C"), *args)
        elif isinstance(argty, types.BaseTuple):
            sig = signature(UsmSharedArrayType(ary.dtype, np.ndim(argty), "C"), *args)
        else:
            raise TypeError("take(%s) not supported for %s" % argty)
        return sig

    def generic_resolve(self, ary, attr):
        # Resolution of other attributes, for record arrays
        if isinstance(ary.dtype, types.Record):
            if attr in ary.dtype.fields:
                return ary.copy(dtype=ary.dtype.typeof(attr), layout="A")


@typing_registry.register_global(nus.as_ndarray)
class DparrayAsNdarray(CallableTemplate):
    def generic(self):
        def typer(arg):
            return types.Array(dtype=arg.dtype, ndim=arg.ndim, layout=arg.layout)

        return typer


@typing_registry.register_global(nus.from_ndarray)
class DparrayFromNdarray(CallableTemplate):
    def generic(self):
        def typer(arg):
            return UsmSharedArrayType(dtype=arg.dtype, ndim=arg.ndim, layout=arg.layout)

        return typer


@lower_registry.lower(nus.as_ndarray, UsmSharedArrayType)
def usmarray_conversion_as(context, builder, sig, args):
    return _array_copy(context, builder, sig, args)


@lower_registry.lower(nus.from_ndarray, types.Array)
def usmarray_conversion_from(context, builder, sig, args):
    return _array_copy(context, builder, sig, args)
