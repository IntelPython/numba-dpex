# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import cached_property

import dpctl.tensor as dpt
import dpnp
from llvmlite import binding as ll
from llvmlite import ir as llvmir
from numba import typeof
from numba.core import cgutils, types, typing, utils
from numba.core.base import BaseContext
from numba.core.callconv import MinimalCallConv
from numba.core.registry import cpu_target
from numba.core.target_extension import GPU, target_registry
from numba.core.types import Array as NpArrayType

from numba_dpex.core.datamodel.models import _init_data_model_manager
from numba_dpex.core.exceptions import UnsupportedKernelArgumentError
from numba_dpex.core.typeconv import to_usm_ndarray
from numba_dpex.core.types import DpnpNdArray, USMNdArray
from numba_dpex.core.utils import get_info_from_suai
from numba_dpex.utils import address_space, calling_conv

from .. import codegen

CC_SPIR_KERNEL = "spir_kernel"
CC_SPIR_FUNC = "spir_func"
VALID_CHARS = re.compile(r"[^a-z0-9]", re.I)
LINK_ATOMIC = 111
LLVM_SPIRV_ARGS = 112


class DpexKernelTypingContext(typing.BaseContext):
    """Custom typing context to support kernel compilation.

    The customized typing context provides two features required to compile
    Python functions decorated by the kernel decorator: An overridden
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
            # We do check here, because IPEX has slow device memory access, so
            # we want to cast it to SUAI before any method call.
            if hasattr(val, "__dlpack__") and not hasattr(
                val, "__sycl_usm_array_interface__"
            ):
                val = dpt.from_dlpack(val)
                suai_attrs = get_info_from_suai(val)
                return to_usm_ndarray(suai_attrs)

            numba_type = typeof(val)
            py_type = type(numba_type)

            if isinstance(numba_type, NpArrayType) and not isinstance(
                numba_type, USMNdArray
            ):
                raise UnsupportedKernelArgumentError(
                    type=str(type(val)), value=val
                )

            # XXX A kernel function has the spir_kernel ABI and requires
            # pointers to have an address space attribute. For this reason, the
            # UsmNdArray type uses a custom data model where the pointers are
            # address space casted to have a SYCL-specific address space value.
            # The DpnpNdArray type on the other hand is meant to be used inside
            # host functions and has Numba's array model as its data model.
            # If the value is a DpnpNdArray then use the ``to_usm_ndarray``
            # function to convert it into a UsmNdArray type rather than passing
            # it to the kernel as a DpnpNdArray. Thus, from a Numba typing
            # perspective dpnp.ndarrays cannot be directly passed to a kernel.
            if py_type is DpnpNdArray:
                suai_attrs = get_info_from_suai(val)
                return to_usm_ndarray(suai_attrs)
        except ValueError:
            # When an array-like kernel argument is not recognized by
            # numba-dpex, this additional check sees if the array-like object
            # implements the __sycl_usm_array_interface__ protocol. For such
            # cases, we treat the object as an UsmNdArray type.
            try:
                suai_attrs = get_info_from_suai(val)
                return to_usm_ndarray(suai_attrs)
            except Exception:
                raise UnsupportedKernelArgumentError(
                    type=str(type(val)), value=val
                )

        return super().resolve_argument_type(val)

    def load_additional_registries(self):
        """Register the OpenCL API and math and other functions."""
        from numba.core.typing import cmathdecl, npydecl

        from ...ocl import mathdecl, ocldecl

        self.install_registry(ocldecl.registry)
        self.install_registry(mathdecl.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(npydecl.registry)


class SyclDevice(GPU):
    """Mark the hardware target as SYCL Device."""

    pass


DPEX_KERNEL_TARGET_NAME = "SyclDevice"

target_registry[DPEX_KERNEL_TARGET_NAME] = SyclDevice


class DpexKernelTargetContext(BaseContext):
    """A target context inheriting Numba's ``BaseContext`` that is customized
    for generating SYCL kernels.

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
        ocl_kernels = cgutils.get_or_insert_named_metadata(
            mod, "opencl.kernels"
        )
        ocl_kernels.add(
            mod.add_metadata(
                [
                    fn,
                    self._gen_arg_addrspace_md(fn),
                    self._gen_arg_type(fn),
                    self._gen_arg_type_qual(fn),
                    self._gen_arg_base_type(fn),
                ],
            )
        )

        # Other metadata
        others = [
            "opencl.used.extensions",
            "opencl.used.optional.core.features",
            "opencl.compiler.options",
        ]

        for name in others:
            nmd = cgutils.get_or_insert_named_metadata(mod, name)
            if not nmd.operands:
                mod.add_metadata([])

    def _generate_kernel_wrapper(self, func, argtypes):
        module = func.module
        arginfo = self.get_arg_packer(argtypes)
        wrapperfnty = llvmir.FunctionType(
            llvmir.VoidType(), arginfo.argument_types
        )
        wrapper_module = self.create_module("dpex.kernel.wrapper")
        wrappername = "dpexPy_{name}".format(name=func.name)
        argtys = list(arginfo.argument_types)
        fnty = llvmir.FunctionType(
            llvmir.IntType(32),
            [self.call_conv.get_return_type(types.pyobject)] + argtys,
        )
        func = llvmir.Function(wrapper_module, fnty, name=func.name)
        func.calling_convention = CC_SPIR_FUNC
        wrapper = llvmir.Function(wrapper_module, wrapperfnty, name=wrappername)
        builder = llvmir.IRBuilder(wrapper.append_basic_block(""))

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

    def __init__(self, typingctx, target=DPEX_KERNEL_TARGET_NAME):
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

    def replace_dpnp_ufunc_with_ocl_intrinsics(self):
        from numba_dpex.ocl.mathimpl import lower_ocl_impl, sig_mapper

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
        from numba_dpex.dpnp_iface import dpnpimpl

        from ... import printimpl
        from ...ocl import mathimpl, oclimpl

        self.insert_func_defn(oclimpl.registry.functions)
        self.insert_func_defn(mathimpl.registry.functions)
        self.insert_func_defn(dpnpimpl.registry.functions)
        self.install_registry(printimpl.registry)
        # Replace dpnp math functions with their OpenCL versions.
        self.replace_dpnp_ufunc_with_ocl_intrinsics()

    @cached_property
    def call_conv(self):
        return DpexCallConv(self)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        return self._target_data

    def mangler(self, name, argtypes, abi_tags=(), uid=None):
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
        ret = super(DpexKernelTargetContext, self).declare_function(
            module, fndesc
        )
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
