# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a SPIR-V code generation-specific target and typing context.
"""

from enum import IntEnum
from functools import cached_property

import dpnp
from llvmlite import binding as ll
from llvmlite import ir as llvmir
from numba.core import cgutils, funcdesc
from numba.core import types as nb_types
from numba.core import typing, utils
from numba.core.base import BaseContext
from numba.core.callconv import MinimalCallConv
from numba.core.target_extension import GPU, target_registry
from numba.core.types.scalars import IntEnumClass
from numba.core.typing import cmathdecl, enumdecl, npydecl

from numba_dpex.core.datamodel.models import _init_data_model_manager
from numba_dpex.core.types import IntEnumLiteral
from numba_dpex.core.typing import dpnpdecl
from numba_dpex.kernel_api.flag_enum import FlagEnum
from numba_dpex.ocl.mathimpl import lower_ocl_impl, sig_mapper
from numba_dpex.utils import address_space, calling_conv

from . import codegen

CC_SPIR_KERNEL = "spir_kernel"
CC_SPIR_FUNC = "spir_func"
LLVM_SPIRV_ARGS = 112


class CompilationMode(IntEnum):
    """Flags used to determine how a function should be compiled by the
    numba_dpex.experimental.dispatcher.KernelDispatcher. Note the functionality
    will be merged into numba_dpex.core.kernel_interface.dispatcher in the
    future.

        KERNEL :         Indicates that the function will be compiled into an
                         LLVM function that has ``spir_kernel`` calling
                         convention and is compiled down to SPIR-V.
                         Additionally, the function cannot return any value and
                         input arguments to the function have to adhere to
                         "compute follows data" to ensure execution queue
                         inference.
        DEVICE_FUNCTION: Indicates that the function will be compiled into an
                         LLVM function that has ``spir_func`` calling convention
                         and will be compiled only into LLVM bitcode.
    """

    KERNEL = 1
    DEVICE_FUNC = 2


class SPIRVTypingContext(typing.BaseContext):
    """Custom typing context to support kernel compilation.

    The customized typing context provides two features required to compile
    Python functions decorated by the kernel decorator: An overridden
    :func:`resolve_argument_type` that changes all ``npytypes.Array`` to
    :class:`numba_depx.core.types.Array`. An overridden
    :func:`load_additional_registries` that registers OpenCL math and other
    functions to the typing context.

    """

    def resolve_value_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a runtime constant.
        ValueError is raised for unsupported types.
        """

        typ = super().resolve_value_type(val)

        if isinstance(typ, IntEnumClass) and issubclass(val, FlagEnum):
            typ = IntEnumLiteral(val)

        return typ

    def resolve_getattr(self, typ, attr):
        """
        Resolve getting the attribute *attr* (a string) on the Numba type.
        The attribute's type is returned, or None if resolution failed.
        """
        retty = None

        if isinstance(typ, IntEnumLiteral):
            try:
                attrval = getattr(typ.literal_value, attr).value
                retty = nb_types.IntegerLiteral(attrval)
            except ValueError:
                pass
        else:
            retty = super().resolve_getattr(typ, attr)
        return retty

    def load_additional_registries(self):
        """Register the OpenCL API and math and other functions."""
        # pylint: disable=import-outside-toplevel
        from numba_dpex.ocl import mathdecl, ocldecl

        self.install_registry(ocldecl.registry)
        self.install_registry(mathdecl.registry)
        self.install_registry(cmathdecl.registry)
        # TODO: https://github.com/IntelPython/numba-dpex/issues/1270
        self.install_registry(npydecl.registry)
        self.install_registry(dpnpdecl.registry)
        self.install_registry(enumdecl.registry)


# pylint: disable=too-few-public-methods
class SPIRVDevice(GPU):
    """Mark the hardware target as device that supports SPIR-V bitcode."""


SPIRV_TARGET_NAME = "spirv"

target_registry[SPIRV_TARGET_NAME] = SPIRVDevice


class SPIRVTargetContext(BaseContext):
    """A target context inheriting Numba's ``BaseContext`` that is customized
    for generating SPIR-V kernels.

    A customized target context for generating SPIR-V kernels. The class defines
    helper functions to generates SPIR-V kernels as LLVM IR using the required
    calling conventions and metadata. The class also registers OpenCL math and
    API functions, helper functions for inserting LLVM address
    space cast instructions, and other functionalities used by the compiler
    to generate SPIR-V kernels.

    """

    implement_powi_as_math_call = True

    def _gen_arg_addrspace_md(self, fn):
        """Generate kernel_arg_addr_space metadata."""
        mod = fn.module
        fnty = fn.type.pointee
        codes = []

        for a in fnty.args:
            if cgutils.is_pointer(a):
                codes.append(address_space.GLOBAL)
            else:
                codes.append(address_space.PRIVATE)

        consts = [llvmir.Constant(llvmir.IntType(32), x) for x in codes]
        name = llvmir.MetaDataString(mod, "kernel_arg_addr_space")
        return mod.add_metadata([name] + consts)

    def _gen_arg_type(self, fn):
        """Generate kernel_arg_type metadata."""
        mod = fn.module
        fnty = fn.type.pointee
        consts = [llvmir.MetaDataString(mod, str(a)) for a in fnty.args]
        name = llvmir.MetaDataString(mod, "kernel_arg_type")
        return mod.add_metadata([name] + consts)

    def _gen_arg_type_qual(self, fn):
        """Generate kernel_arg_type_qual metadata."""
        mod = fn.module
        fnty = fn.type.pointee
        consts = [llvmir.MetaDataString(mod, "") for _ in fnty.args]
        name = llvmir.MetaDataString(mod, "kernel_arg_type_qual")
        return mod.add_metadata([name] + consts)

    def _gen_arg_base_type(self, fn):
        """Generate kernel_arg_base_type metadata."""
        mod = fn.module
        fnty = fn.type.pointee
        consts = [llvmir.MetaDataString(mod, str(a)) for a in fnty.args]
        name = llvmir.MetaDataString(mod, "kernel_arg_base_type")
        return mod.add_metadata([name] + consts)

    def _finalize_kernel_wrapper_module(self, fn):
        """Add metadata and calling convention to the wrapper function.

        The helper function adds function metadata to the wrapper function and
        also module level metadata to the LLVM module containing the wrapper.
        We also make sure the wrapper function has ``spir_kernel`` calling
        convention, without which the function cannot be used as a kernel.

        Args:
            fn: LLVM function representing the "kernel" wrapper function.

        """
        # Set norecurse
        fn.attributes.add("norecurse")
        # Set SPIR kernel calling convention
        fn.calling_convention = CC_SPIR_KERNEL

    def _generate_spir_kernel_wrapper(self, func, argtypes):
        module = func.module
        arginfo = self.get_arg_packer(argtypes)
        wrapperfnty = llvmir.FunctionType(
            llvmir.VoidType(), arginfo.argument_types
        )
        wrapper_module = self._internal_codegen.create_empty_spirv_module(
            "dpex.kernel.wrapper"
        )
        wrappername = func.name.replace("dpex_fn", "dpex_kernel")
        argtys = list(arginfo.argument_types)
        fnty = llvmir.FunctionType(
            llvmir.IntType(32),
            [self.call_conv.get_return_type(nb_types.pyobject)] + argtys,
        )
        func = llvmir.Function(wrapper_module, fnty, name=func.name)
        func.calling_convention = CC_SPIR_FUNC
        wrapper = llvmir.Function(wrapper_module, wrapperfnty, name=wrappername)
        builder = llvmir.IRBuilder(wrapper.append_basic_block("entry"))

        callargs = arginfo.from_arguments(builder, wrapper.args)

        # XXX handle error status
        self.call_conv.call_function(
            builder, func, nb_types.void, argtypes, callargs
        )
        builder.ret_void()

        self._finalize_kernel_wrapper_module(wrapper)

        # Link the spir_func module to the wrapper module
        module.link_in(ll.parse_assembly(str(wrapper_module)))
        # Make sure the spir_func has internal linkage to be inlinable.
        func.linkage = "internal"
        wrapper = module.get_function(wrapper.name)
        module.get_function(func.name).linkage = "internal"
        return wrapper

    def __init__(self, typingctx, target=SPIRV_TARGET_NAME):
        super().__init__(typingctx, target)

    def init(self):
        """Called by the super().__init__ constructor to initalize the child
        class.
        """
        # pylint: disable=import-outside-toplevel
        from numba_dpex.dpnp_iface.dpnp_ufunc_db import _lazy_init_dpnp_db

        self._internal_codegen = codegen.JITSPIRVCodegen("numba_dpex.kernel")
        self._target_data = ll.create_target_data(
            codegen.SPIR_DATA_LAYOUT[utils.MACHINE_BITS]
        )

        # Override data model manager to SPIR model
        self.data_model_manager = _init_data_model_manager()
        self.extra_compile_options = {}

        _lazy_init_dpnp_db()

        # we need to import it after, because before init it is None and
        # variable is passed by value
        from numba_dpex.dpnp_iface.dpnp_ufunc_db import _dpnp_ufunc_db

        self.ufunc_db = _dpnp_ufunc_db

    def get_getattr(self, typ, attr):
        """
        Overrides the get_getattr function to provide an implementation for
        getattr call on an IntegerEnumLiteral type.
        """

        if isinstance(typ, IntEnumLiteral):
            #  pylint: disable=W0613
            def enum_literal_getattr_imp(context, builder, typ, val, attr):
                enum_attr_value = getattr(typ.literal_value, attr).value
                return llvmir.Constant(llvmir.IntType(64), enum_attr_value)

            return enum_literal_getattr_imp

        return super().get_getattr(typ, attr)

    def create_module(self, name):
        return self._internal_codegen.create_empty_spirv_module(name)

    def replace_dpnp_ufunc_with_ocl_intrinsics(self):
        """Replaces the implementation in the ufunc_db for specific math
        functions for which a SPIR-V intrinsic should be used.
        """
        ufuncs = [
            ("fabs", dpnp.fabs),
            ("exp", dpnp.exp),
            ("log", dpnp.log),
            ("log10", dpnp.log10),
            ("expm1", dpnp.expm1),
            ("log1p", dpnp.log1p),
            ("sqrt", dpnp.sqrt),
            ("sin", dpnp.sin),
            ("cos", dpnp.cos),
            ("tan", dpnp.tan),
            ("asin", dpnp.arcsin),
            ("acos", dpnp.arccos),
            ("atan", dpnp.arctan),
            ("atan2", dpnp.arctan2),
            ("sinh", dpnp.sinh),
            ("cosh", dpnp.cosh),
            ("tanh", dpnp.tanh),
            ("asinh", dpnp.arcsinh),
            ("acosh", dpnp.arccosh),
            ("atanh", dpnp.arctanh),
            ("floor", dpnp.floor),
            ("ceil", dpnp.ceil),
            ("trunc", dpnp.trunc),
            ("hypot", dpnp.hypot),
            ("exp2", dpnp.exp2),
            ("log2", dpnp.log2),
        ]

        for name, ufunc in ufuncs:
            for sig in self.ufunc_db[ufunc].keys():
                if (
                    sig in sig_mapper
                    and (name, sig_mapper[sig]) in lower_ocl_impl
                ):
                    self.ufunc_db[ufunc][sig] = lower_ocl_impl[
                        (name, sig_mapper[sig])
                    ]

    def load_additional_registries(self):
        """Register OpenCL functions into numba_depx's target context.

        To make sure we are calling supported OpenCL math functions, we replace
        the dpnp functions that default to NUMBA's NumPy ufunc with OpenCL
        intrinsics that are equivalent to those functions. The replacement is
        done after the OpenCL functions have been registered into the
        target context.

        """
        # pylint: disable=import-outside-toplevel
        from numba_dpex import printimpl
        from numba_dpex.dpnp_iface import dpnpimpl
        from numba_dpex.experimental._kernel_dpcpp_spirv_overloads._registry import (
            registry as spirv_registry,
        )
        from numba_dpex.ocl import mathimpl, oclimpl

        self.insert_func_defn(oclimpl.registry.functions)
        self.insert_func_defn(mathimpl.registry.functions)
        self.insert_func_defn(dpnpimpl.registry.functions)
        self.install_registry(printimpl.registry)
        self.install_registry(spirv_registry)
        # Replace dpnp math functions with their OpenCL versions.
        self.replace_dpnp_ufunc_with_ocl_intrinsics()

    @cached_property
    def call_conv(self):
        """
        Return the CallConv object used by the SPIRVTargetContext.
        """
        return SPIRVCallConv(self)

    def codegen(self):
        """Return the CodeGen object used by the SPIRVTargetContext."""
        return self._internal_codegen

    @property
    def target_data(self):
        return self._target_data

    def mangler(self, name, types, *, abi_tags=(), uid=None):
        """
        Generates a name for a function by appending \"dpex_fn\" to the
        name of the function before calling Numba's default function name
        mangler."""
        return funcdesc.default_mangler(
            name + "dpex_fn", types, abi_tags=abi_tags, uid=uid
        )

    def prepare_spir_kernel(self, func, argtypes):
        """Generates a wrapper function with \"spir_kernel\" calling conv that
        calls the compiled \"spir_func\" generated by numba_dpex for a kernel
        decorated function.
        """
        func.linkage = "linkonce_odr"
        func.module.data_layout = codegen.SPIR_DATA_LAYOUT[self.address_size]
        wrapper = self._generate_spir_kernel_wrapper(func, argtypes)
        return wrapper

    def set_spir_func_calling_conv(self, func):
        """Sets the calling convetion of the provided LLVM func to spir_func"""
        # Adapt to SPIR
        func.calling_convention = CC_SPIR_FUNC
        func.linkage = "linkonce_odr"
        return func

    def declare_function(self, module, fndesc):
        """Create the LLVM function from a ``numba_dpex.kernel`` decorated
        function.

        Args:
            module (llvmlite.ir.Module) : The LLVM module into which
                the kernel function will be inserted.
            fndesc (numba.core.funcdesc.PythonFunctionDescriptor) : The
                signature of the function.

        Returns:
            llvmlite.ir.values.Function: The reference to the LLVM Function
                that was inserted into the module.

        """
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        fn = cgutils.get_or_insert_function(
            module, fnty, name=fndesc.mangled_name
        )
        if not self.enable_debuginfo:
            fn.attributes.add("alwaysinline")
        ret = super().declare_function(module, fndesc)
        ret.calling_convention = calling_conv.CC_SPIR_FUNC
        return ret

    def insert_const_string(self, mod, string):
        """Create a global string from the passed in string argument and return
        a void* in the GENERIC address space pointing to that string.

        Args:
            mod: LLVM module where the global string value is to be inserted.
            string: A Python string that will be converted to a global constant
            string and inserted into the module.

        Returns: A LLVM Constant pointing to the global string value inserted
        into the module.

        """
        text = cgutils.make_bytearray(string.encode("utf-8") + b"\x00")

        name = "$".join(["__conststring__", self.mangler(string, ["str"])])

        # Try to reuse existing global
        gv = mod.globals.get(name)
        if gv is None:
            # Not defined yet
            gv = cgutils.add_global_variable(
                mod, text.type, name=name, addrspace=address_space.GENERIC
            )
            gv.linkage = "internal"
            gv.global_constant = True
            gv.initializer = text

        # Cast to a i8* pointer
        charty = gv.type.pointee.element
        return gv.bitcast(charty.as_pointer(address_space.GENERIC))

    def addrspacecast(self, builder, src, addrspace):
        """Insert an LLVM addressspace cast instruction into the module.

        FIXME: Move this function into utils.

        """
        ptras = llvmir.PointerType(src.type.pointee, addrspace=addrspace)
        return builder.addrspacecast(src, ptras)

    def get_ufunc_info(self, ufunc_key):
        return self.ufunc_db[ufunc_key]

    def populate_array(self, arr, **kwargs):
        """
        Populate array structure.
        """
        # pylint: disable=import-outside-toplevel
        from numba_dpex.core.kernel_interface import arrayobj

        return arrayobj.populate_array(arr, **kwargs)

    def get_executable(self, func, fndesc, env):
        """Not implemented for SPIRVTargetContext"""
        raise NotImplementedError("Not implemented for SPIRVTargetContext")


class SPIRVCallConv(MinimalCallConv):
    """Custom calling convention class used by numba-dpex.

    numba_dpex's calling convention derives from
    :class:`numba.core.callconv import MinimalCallConv`. The
    :class:`SPIRVCallConv` overrides :func:`call_function`.

    """

    # pylint:disable=too-many-arguments
    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """Call the Numba-compiled *callee*."""
        assert env is None
        retty = callee.args[0].type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(cgutils.get_null_value(retty), retvaltmp)
        arginfo = self.context.get_arg_packer(argtys)
        args = arginfo.as_arguments(builder, args)
        realargs = [retvaltmp] + list(args)
        code = builder.call(callee, realargs)
        status = self._get_return_status(builder, code)
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out
