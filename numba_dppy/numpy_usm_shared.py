import numpy as np
from inspect import getmembers, isfunction, isclass, isbuiltin
from numbers import Number
import numba
from types import FunctionType as ftype, BuiltinFunctionType as bftype
from numba import types
from numba.extending import typeof_impl, register_model, type_callable, lower_builtin
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
from numba.core.typing.templates import CallableTemplate
from numba.np.arrayobj import _array_copy

from dpctl.dptensor.numpy_usm_shared import ndarray, functions_list


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

# functions_list = [o[0] for o in getmembers(np) if isfunction(o[1]) or isbuiltin(o[1])]
# class_list = [o for o in getmembers(np) if isclass(o[1])]

# Register the helper function in dppl_rt so that we can insert calls to them via llvmlite.
for py_name, c_address in numba_dppy._dppy_rt.c_helpers.items():
    llb.add_symbol(py_name, c_address)


# This class creates a type in Numba.
class UsmSharedArrayType(types.Array):
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
register_model(UsmSharedArrayType)(numba.core.datamodel.models.ArrayModel)

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


registered = False


def numba_register():
    global registered
    if not registered:
        registered = True
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

    # For all Numpy identifiers that have been registered for typing in Numba...
    # this registry contains functions, getattrs, setattrs, casts and constants...need to do them all? FIX FIX FIX
    for ig in lower_registry.functions:
        impl, func, types = ig
        # If it is a Numpy function...
        if isinstance(func, ftype):
            if func.__module__ == np.__name__:
                # If we have overloaded that function in the usmarray module (always True right now)...
                if func.__name__ in functions_list:
                    todo.append(ig)
        if isinstance(func, bftype):
            if func.__module__ == np.__name__:
                # If we have overloaded that function in the usmarray module (always True right now)...
                if func.__name__ in functions_list:
                    todo.append(ig)

    for lg in lower_registry.getattrs:
        func, attr, types = lg
        types_with_usmarray = types_replace_array(types)
        if UsmSharedArrayType in types_with_usmarray:
            dprint(
                "lower_getattr:", func, type(func), attr, type(attr), types, type(types)
            )
            todo_getattr.append((func, attr, types_with_usmarray))

    for lg in todo_getattr:
        lower_registry.getattrs.append(lg)

    cur_mod = importlib.import_module(__name__)
    for impl, func, types in todo + todo_builtin:
        usmarray_func = eval(func.__name__)
        dprint(
            "need to re-register lowerer for usmarray", impl, func, types, usmarray_func
        )
        new_impl = copy_func_for_usmarray(impl, cur_mod)
        lower_registry.functions.append((new_impl, usmarray_func, types))


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
        # If it is a Numpy function...
        if isinstance(val, (ftype, bftype)):
            # If we have overloaded that function in the usmarray module (always True right now)...
            if val.__name__ in functions_list:
                todo.append(ig)
        if isinstance(val, type):
            todo_classes.append(ig)

    for tgetattr in templates_registry.attributes:
        if tgetattr.key == types.Array:
            todo_getattr.append(tgetattr)

    for val, typ in todo:
        assert len(typ.templates) == 1
        # template is the typing class to invoke generic() upon.
        template = typ.templates[0]
        dpval = eval(val.__name__)
        dprint("need to re-register for usmarray", val, typ, typ.typing_key)
        """
        if debug:
            print("--------------------------------------------------------------")
            print("need to re-register for usmarray", val, typ, typ.typing_key)
            print("val:", val, type(val), "dir val", dir(val))
            print("typ:", typ, type(typ), "dir typ", dir(typ))
            print("typing key:", typ.typing_key)
            print("name:", typ.name)
            print("key:", typ.key)
            print("templates:", typ.templates)
            print("template:", template, type(template))
            print("dpval:", dpval, type(dpval))
            print("--------------------------------------------------------------")
        """

        class_name = "DparrayTemplate_" + val.__name__

        @classmethod
        def set_key_original(cls, key, original):
            cls.key = key
            cls.original = original

        def generic_impl(self):
            original_typer = self.__class__.original.generic(self.__class__.original)
            ot_argspec = inspect.getfullargspec(original_typer)
            # print("ot_argspec:", ot_argspec)
            astr = argspec_to_string(ot_argspec)
            # print("astr:", astr)

            typer_func = """def typer({}):
                                original_res = original_typer({})
                                #print("original_res:", original_res)
                                if isinstance(original_res, types.Array):
                                    return UsmSharedArrayType(dtype=original_res.dtype, ndim=original_res.ndim, layout=original_res.layout)

                                return original_res""".format(
                astr, ",".join(ot_argspec.args)
            )

            # print("typer_func:", typer_func)

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

            # print("exec_res:", exec_res)
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
    for tgetattr in todo_getattr:
        class_name = tgetattr.__name__ + "_usmarray"
        dprint("tgetattr:", tgetattr, type(tgetattr), class_name)

        @classmethod
        def set_key(cls, key):
            cls.key = key

        def getattr_impl(self, attr):
            if attr.startswith("resolve_"):
                # print("getattr_impl starts with resolve_:", self, type(self), attr)
                def wrapper(*args, **kwargs):
                    attr_res = tgetattr.__getattribute__(self, attr)(*args, **kwargs)
                    if isinstance(attr_res, types.Array):
                        return UsmSharedArrayType(
                            dtype=attr_res.dtype,
                            ndim=attr_res.ndim,
                            layout=attr_res.layout,
                        )

                return wrapper
            else:
                return tgetattr.__getattribute__(self, attr)

        new_usmarray_template = type(
            class_name,
            (tgetattr,),
            {"set_class_vars": set_key, "__getattribute__": getattr_impl},
        )

        new_usmarray_template.set_class_vars(UsmSharedArrayType)
        templates_registry.register_attr(new_usmarray_template)


def from_ndarray(x):
    return copy(x)


def as_ndarray(x):
    return np.copy(x)


@typing_registry.register_global(as_ndarray)
class DparrayAsNdarray(CallableTemplate):
    def generic(self):
        def typer(arg):
            return types.Array(dtype=arg.dtype, ndim=arg.ndim, layout=arg.layout)

        return typer


@typing_registry.register_global(from_ndarray)
class DparrayFromNdarray(CallableTemplate):
    def generic(self):
        def typer(arg):
            return UsmSharedArrayType(dtype=arg.dtype, ndim=arg.ndim, layout=arg.layout)

        return typer


@lower_registry.lower(as_ndarray, UsmSharedArrayType)
def usmarray_conversion_as(context, builder, sig, args):
    return _array_copy(context, builder, sig, args)


@lower_registry.lower(from_ndarray, types.Array)
def usmarray_conversion_from(context, builder, sig, args):
    return _array_copy(context, builder, sig, args)
