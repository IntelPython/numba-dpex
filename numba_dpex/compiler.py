# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from types import FunctionType

import dpctl.program as dpctl_prog
from numba.core import compiler, ir, types
from numba.core.compiler_lock import global_compiler_lock

from numba_dpex import config
from numba_dpex.core.compiler import Compiler
from numba_dpex.core.exceptions import KernelHasReturnValueError
from numba_dpex.core.types import Array

from . import spirv_generator


@global_compiler_lock
def compile_with_depx(pyfunc, return_type, args, is_kernel, debug=None):
    """
    Compiles the function using the dpex compiler pipeline and returns the
    compiled result.

    Args:
        pyfunc: The Python function to be compiled.
        return_type: The Numba type of the return value.
        args: The list of arguments sent to the Python function.
        is_kernel (bool): Indicates whether the function is decorated
        with @numba_depx.kernel or not.
        debug (bool): Flag to turn debug mode ON/OFF.

    Returns:
        cres: Compiled result.

    Raises:
        TypeError: @numba_depx.kernel does not allow users to return any
            value. TypeError is raised when users do.

    """
    # First compilation will trigger the initialization of the backend.
    from .core.descriptor import dpex_target

    typingctx = dpex_target.typing_context
    targetctx = dpex_target.target_context

    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.debuginfo = config.DEBUGINFO_DEFAULT
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.nrt = False

    if debug is not None:
        flags.debuginfo = debug

    # Run compilation pipeline
    if isinstance(pyfunc, FunctionType):
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=Compiler,
        )
    elif isinstance(pyfunc, ir.FunctionIR):
        cres = compiler.compile_ir(
            typingctx=typingctx,
            targetctx=targetctx,
            func_ir=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=Compiler,
        )
    else:
        assert 0

    if (
        is_kernel
        and cres.signature.return_type is not None
        and cres.signature.return_type != types.void
    ):
        raise KernelHasReturnValueError(
            kernel_name=pyfunc.__name__, return_type=cres.signature.return_type
        )

    # Linking depending libraries
    library = cres.library
    library.finalize()

    return cres


def compile_kernel_parfor(
    sycl_queue, func_ir, args, args_with_addrspaces, debug=None
):
    # We only accept numba_dpex.core.types.Array type
    for arg in args_with_addrspaces:
        if isinstance(arg, types.npytypes.Array) and not isinstance(arg, Array):
            raise TypeError(
                "Only numba_dpex.core.types.Array objects are supported as "
                + "kernel arguments. Received %s" % (type(arg))
            )
    if config.DEBUG:
        print("compile_kernel_parfor", args)
        for a in args_with_addrspaces:
            print(a, type(a))
            if isinstance(a, types.npytypes.Array):
                print("addrspace:", a.addrspace)

    cres = compile_with_depx(
        pyfunc=func_ir,
        return_type=None,
        args=args_with_addrspaces,
        is_kernel=True,
        debug=debug,
    )
    func = cres.library.get_function(cres.fndesc.llvm_func_name)

    if config.DEBUG:
        print("compile_kernel_parfor signature", cres.signature.args)
        for a in cres.signature.args:
            print(a, type(a))

    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = Kernel(
        context=cres.target_context,
        sycl_queue=sycl_queue,
        llvm_module=kernel.module,
        name=kernel.name,
        argtypes=args_with_addrspaces,
    )

    return oclkern


class KernelBase(object):
    """Define interface for configurable kernels"""

    def __init__(self):
        self.global_size = []
        self.local_size = []
        self.sycl_queue = None


class Kernel(KernelBase):
    """
    A OCL kernel object
    """

    def __init__(
        self,
        context,
        sycl_queue,
        llvm_module,
        name,
        argtypes,
        ordered_arg_access_types=None,
    ):
        super(Kernel, self).__init__()
        self._llvm_module = llvm_module
        self.assembly = self.binary = llvm_module.__str__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.ordered_arg_access_types = ordered_arg_access_types
        self._argloc = []
        self.sycl_queue = sycl_queue
        self.context = context

        dpctl_create_program_from_spirv_flags = []
        # First-time compilation using SPIRV-Tools
        if config.DEBUG:
            with open("llvm_kernel.ll", "w") as f:
                f.write(self.binary)

        if config.DEBUG or config.OPT == 0:
            # if debug is ON we need to pass additional
            # flags to igc.
            dpctl_create_program_from_spirv_flags = ["-g", "-cl-opt-disable"]

        self.spirv_bc = spirv_generator.llvm_to_spirv(
            self.context, self.assembly, self._llvm_module.as_bitcode()
        )

        # create a program
        self.program = dpctl_prog.create_program_from_spirv(
            self.sycl_queue,
            self.spirv_bc,
            " ".join(dpctl_create_program_from_spirv_flags),
        )
        #  create a kernel
        self.kernel = self.program.get_sycl_kernel(self.entry_name)
