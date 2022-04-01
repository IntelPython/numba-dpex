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

import re

import numpy as np
from llvmlite import binding as ll
from llvmlite import ir as llvmir
from llvmlite.llvmpy import core as lc
from numba import typeof
from numba.core import cgutils, datamodel, types, typing, utils
from numba.core.base import BaseContext
from numba.core.callconv import MinimalCallConv
from numba.core.registry import cpu_target
from numba.core.target_extension import GPU, target_registry
from numba.core.utils import cached_property

from numba_dppy.core.types import Array, ArrayModel
from numba_dppy.utils import (
    address_space,
    calling_conv,
    has_usm_memory,
    npytypes_array_to_dpex_array,
    suai_to_dpex_array,
)

from . import codegen

CC_SPIR_KERNEL = "spir_kernel"
CC_SPIR_FUNC = "spir_func"
VALID_CHARS = re.compile(r"[^a-z0-9]", re.I)
LINK_ATOMIC = 111
LLVM_SPIRV_ARGS = 112


class DpexTypingContext(typing.BaseContext):
    """A typing context inheriting Numba's ``BaseContext`` to support
    dpex-specific data types.

    :class:`DpexTypingContext` is a customized typing context that inherits from
    Numba's ``typing.BaseContext`` class. We add two specific functionalities to
    the basic Numba typing context features: An overridden
    :func:`resolve_argument_type` that changes all ``npytypes.Array`` to
    :class:`numba_depx.core.types.Array`. An overridden
    :func:`load_additional_registries` that registers OpenCL math and other
    functions to the typing context.

    """

    def resolve_argument_type(self, val):
        """Return the Numba type of a Python value used as a function argument.

        Overrides the implementation of ``numba.core.typing.BaseContext`` to
        handle the special case of ``numba.core.types.npytypes.Array``. Whenever
        a NumPy ndarray argument is encountered as an argument to a ``kernel``
        function, it is converted to a ``numba_dpex.core.types.Array`` type.

        Args:
            val : A Python value that is passed as an argument to a ``kernel``
                  function.

        Returns: The Numba type corresponding to the Python value.

        Raises:
            ValueError: If the type of the Python value is not supported.

        """
        try:
            _type = type(typeof(val))
        except ValueError:
            # For arbitrary array that is not recognized by Numba,
            # we will end up in this path. We check if the array
            # has __sycl_usm_array_interface__ attribute. If yes,
            # we create the necessary Numba type to represent it
            # and send it back.
            if has_usm_memory(val) is not None:
                return suai_to_dpex_array(val)

        if _type is types.npytypes.Array:
            # Convert npytypes.Array to numba_dpex.core.types.Array
            return npytypes_array_to_dpex_array(typeof(val))
        else:
            return super().resolve_argument_type(val)

    def load_additional_registries(self):
        """Register the OpenCL API and math and other functions."""
        from numba.core.typing import cmathdecl, npydecl

        from .ocl import mathdecl, ocldecl

        self.install_registry(ocldecl.registry)
        self.install_registry(mathdecl.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(npydecl.registry)


class GenericPointerModel(datamodel.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        adrsp = (
            fe_type.addrspace
            if fe_type.addrspace is not None
            else address_space.GLOBAL
        )
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer(adrsp)
        super(GenericPointerModel, self).__init__(dmm, fe_type, be_type)


def _init_data_model_manager():
    dmm = datamodel.default_manager.copy()
    dmm.register(types.CPointer, GenericPointerModel)
    dmm.register(Array, ArrayModel)
    return dmm


spirv_data_model_manager = _init_data_model_manager()


class SyclDevice(GPU):
    """Mark the hardware target as SYCL Device."""

    pass


DPEX_TARGET_NAME = "SyclDevice"

target_registry[DPEX_TARGET_NAME] = SyclDevice

import numba_dppy.offload_dispatcher


class DpexTargetContext(BaseContext):
    """A target context inheriting Numba's ``BaseContext`` that is customized
    for generating SYCL kernels.

    :class:`DpexTargetContext` is a customized target context that inherits from
    Numba's ``numba.core.base.BaseContext`` class. The class defines helper
    functions to mark LLVM functions as SPIR-V kernels. The class also registers
    OpenCL math and API functions, helper functions for inserting LLVM address
    space cast instructions, and other functionalities used by dpex compiler
    passes.

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

        consts = [lc.Constant.int(lc.Type.int(), x) for x in codes]
        name = lc.MetaDataString.get(mod, "kernel_arg_addr_space")
        return lc.MetaData.get(mod, [name] + consts)

    def _gen_arg_access_qual_md(self, fn):
        """Generate kernel_arg_access_qual metadata."""
        mod = fn.module
        consts = [lc.MetaDataString.get(mod, "none")] * len(fn.args)
        name = lc.MetaDataString.get(mod, "kernel_arg_access_qual")
        return lc.MetaData.get(mod, [name] + consts)

    def _gen_arg_type(self, fn):
        """Generate kernel_arg_type metadata."""
        mod = fn.module
        fnty = fn.type.pointee
        consts = [lc.MetaDataString.get(mod, str(a)) for a in fnty.args]
        name = lc.MetaDataString.get(mod, "kernel_arg_type")
        return lc.MetaData.get(mod, [name] + consts)

    def _gen_arg_type_qual(self, fn):
        """Generate kernel_arg_type_qual metadata."""
        mod = fn.module
        fnty = fn.type.pointee
        consts = [lc.MetaDataString.get(mod, "") for _ in fnty.args]
        name = lc.MetaDataString.get(mod, "kernel_arg_type_qual")
        return lc.MetaData.get(mod, [name] + consts)

    def _gen_arg_base_type(self, fn):
        """Generate kernel_arg_base_type metadata."""
        mod = fn.module
        fnty = fn.type.pointee
        consts = [lc.MetaDataString.get(mod, str(a)) for a in fnty.args]
        name = lc.MetaDataString.get(mod, "kernel_arg_base_type")
        return lc.MetaData.get(mod, [name] + consts)

    def _finalize_wrapper_module(self, fn):
        """Add metadata and calling convention to the wrapper function.

        The helper function adds function metadata to the wrapper function and
        also module level metadata to the LLVM module containing the wrapper.
        We also make sure the wrapper function has ``spir_kernel`` calling
        convention, without which the function cannot be used as a kernel.

        Args:
            fn: LLVM function representing the "kernel" wrapper function.

        """
        mod = fn.module
        # Set norecurse
        fn.attributes.add("norecurse")
        # Set SPIR kernel calling convention
        fn.calling_convention = CC_SPIR_KERNEL

        # Mark kernels
        ocl_kernels = mod.get_or_insert_named_metadata("opencl.kernels")
        ocl_kernels.add(
            lc.MetaData.get(
                mod,
                [
                    fn,
                    self._gen_arg_addrspace_md(fn),
                    self._gen_arg_access_qual_md(fn),
                    self._gen_arg_type(fn),
                    self._gen_arg_type_qual(fn),
                    self._gen_arg_base_type(fn),
                ],
            )
        )

        # Other metadata
        empty_md = lc.MetaData.get(mod, ())
        others = [
            "opencl.used.extensions",
            "opencl.used.optional.core.features",
            "opencl.compiler.options",
        ]

        for name in others:
            nmd = mod.get_or_insert_named_metadata(name)
            if not nmd.operands:
                nmd.add(empty_md)

    def _generate_kernel_wrapper(self, func, argtypes):
        module = func.module
        arginfo = self.get_arg_packer(argtypes)
        wrapperfnty = lc.Type.function(lc.Type.void(), arginfo.argument_types)
        wrapper_module = self.create_module("dpex.kernel.wrapper")
        wrappername = "dpexPy_{name}".format(name=func.name)
        argtys = list(arginfo.argument_types)
        fnty = lc.Type.function(
            lc.Type.int(),
            [self.call_conv.get_return_type(types.pyobject)] + argtys,
        )
        func = wrapper_module.add_function(fnty, name=func.name)
        func.calling_convention = CC_SPIR_FUNC
        wrapper = wrapper_module.add_function(wrapperfnty, name=wrappername)
        builder = lc.Builder(wrapper.append_basic_block(""))

        callargs = arginfo.from_arguments(builder, wrapper.args)

        # XXX handle error status
        status, _ = self.call_conv.call_function(
            builder, func, types.void, argtypes, callargs
        )
        builder.ret_void()

        self._finalize_wrapper_module(wrapper)

        # Link the spir_func module to the wrapper module
        module.link_in(ll.parse_assembly(str(wrapper_module)))
        # Make sure the spir_func has internal linkage to be inlinable.
        func.linkage = "internal"
        wrapper = module.get_function(wrapper.name)
        module.get_function(func.name).linkage = "internal"
        return wrapper

    def __init__(self, typingctx, target=DPEX_TARGET_NAME):
        super().__init__(typingctx, target)

    def init(self):
        self._internal_codegen = codegen.JITSPIRVCodegen("numba_dpex.jit")
        self._target_data = ll.create_target_data(
            codegen.SPIR_DATA_LAYOUT[utils.MACHINE_BITS]
        )
        # Override data model manager to SPIR model
        import numba.cpython.unicode

        self.data_model_manager = _init_data_model_manager()
        self.extra_compile_options = dict()

        import copy

        from numba.np.ufunc_db import _lazy_init_db

        _lazy_init_db()
        from numba.np.ufunc_db import _ufunc_db as ufunc_db

        self.ufunc_db = copy.deepcopy(ufunc_db)
        self.cpu_context = cpu_target.target_context

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def replace_numpy_ufunc_with_opencl_supported_functions(self):
        from numba_dppy.ocl.mathimpl import lower_ocl_impl, sig_mapper

        ufuncs = [
            ("fabs", np.fabs),
            ("exp", np.exp),
            ("log", np.log),
            ("log10", np.log10),
            ("expm1", np.expm1),
            ("log1p", np.log1p),
            ("sqrt", np.sqrt),
            ("sin", np.sin),
            ("cos", np.cos),
            ("tan", np.tan),
            ("asin", np.arcsin),
            ("acos", np.arccos),
            ("atan", np.arctan),
            ("atan2", np.arctan2),
            ("sinh", np.sinh),
            ("cosh", np.cosh),
            ("tanh", np.tanh),
            ("asinh", np.arcsinh),
            ("acosh", np.arccosh),
            ("atanh", np.arctanh),
            ("ldexp", np.ldexp),
            ("floor", np.floor),
            ("ceil", np.ceil),
            ("trunc", np.trunc),
            ("hypot", np.hypot),
            ("exp2", np.exp2),
            ("log2", np.log2),
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

        To make sure we are calling supported OpenCL math functions, we
        replace some of NUMBA's NumPy ufunc with OpenCL versions of those
        functions. The replacement is done after the OpenCL functions have
        been registered into the target context.

        """
        from numba.np import npyimpl

        from . import printimpl
        from .ocl import mathimpl, oclimpl

        self.insert_func_defn(oclimpl.registry.functions)
        self.insert_func_defn(mathimpl.registry.functions)
        self.insert_func_defn(npyimpl.registry.functions)
        self.install_registry(printimpl.registry)
        # Replace NumPy functions with their OpenCL versions.
        self.replace_numpy_ufunc_with_opencl_supported_functions()

    @cached_property
    def call_conv(self):
        return DpexCallConv(self)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        return self._target_data

    def mangler(self, name, argtypes, abi_tags=()):
        def repl(m):
            ch = m.group(0)
            return "_%X_" % ord(ch)

        qualified = name + "." + ".".join(str(a) for a in argtypes)
        mangled = VALID_CHARS.sub(repl, qualified)
        return "dpex_py_devfn_" + mangled

    def prepare_ocl_kernel(self, func, argtypes):
        module = func.module
        func.linkage = "linkonce_odr"
        module.data_layout = codegen.SPIR_DATA_LAYOUT[self.address_size]
        wrapper = self._generate_kernel_wrapper(func, argtypes)
        return wrapper

    def mark_ocl_device(self, func):
        # Adapt to SPIR
        func.calling_convention = CC_SPIR_FUNC
        func.linkage = "linkonce_odr"
        return func

    def declare_function(self, module, fndesc):
        """Create the LLVM function from a ``numba_dpex.kernel`` decorated
        function.

        Args:
            module (llvmlite.llvmpy.core.Module) : The LLVM module into which
                the kernel function will be inserted.
            fndesc (numba.core.funcdesc.PythonFunctionDescriptor) : The
                signature of the function.

        Returns:
            llvmlite.ir.values.Function: The reference to the LLVM Function
                that was inserted into the module.

        """
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        fn = module.get_or_insert_function(fnty, name=fndesc.mangled_name)
        if not self.enable_debuginfo:
            fn.attributes.add("alwaysinline")
        ret = super(DpexTargetContext, self).declare_function(module, fndesc)
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

    # Overrides
    def get_ufunc_info(self, ufunc_key):
        return self.ufunc_db[ufunc_key]


class DpexCallConv(MinimalCallConv):
    """Custom calling convention class used by numba-dpex.

    numba_dpex's calling convention derives from
    :class:`numba.core.callconv import MinimalCallConv`. The
    :class:`DpexCallConv` overriddes :func:`call_function`.

    """

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
