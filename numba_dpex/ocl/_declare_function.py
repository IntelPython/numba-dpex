# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from numba.core import cgutils, types

from numba_dpex.core.utils.itanium_mangler import mangle_c

# -----------------------------------------------------------------------------


def _declare_function(context, builder, name, sig, cargs, mangler=mangle_c):
    """Insert declaration for a opencl builtin function.
    Uses the Itanium mangler.

    Args
    ----
    context: target context

    builder: llvm builder

    name: str
        symbol name

    sig: signature
        function signature of the symbol being declared

    cargs: sequence of str
        C type names for the arguments

    mangler: a mangler function
        function to use to mangle the symbol

    """
    mod = builder.module
    if sig.return_type == types.void:
        llretty = llvmir.VoidType()
    else:
        llretty = context.get_value_type(sig.return_type)
    llargs = [context.get_value_type(t) for t in sig.args]
    fnty = llvmir.FunctionType(llretty, llargs)
    mangled = mangler(name, cargs)
    fn = cgutils.get_or_insert_function(mod, fnty, mangled)
    from numba_dpex import spirv_kernel_target

    fn.calling_convention = spirv_kernel_target.CC_SPIR_FUNC
    return fn
