# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
A wrapper to call dpcpp's llvm-spirv tool to generate a SPIR-V binary from
a numba-dpex generated LLVM IR module.
"""

import os
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output

from numba_dpex.core import config
from numba_dpex.core.exceptions import InternalError
from numba_dpex.kernel_api_impl.spirv.target import LLVM_SPIRV_ARGS

try:
    import dpcpp_llvm_spirv as dls
except ImportError as err:
    raise ImportError("Cannot import dpcpp-llvm-spirv package") from err


def run_cmd(args, error_message=None):
    """
    Helper to run an external command and provide a meaningful error message.
    """
    try:
        check_output(
            args,
            stderr=STDOUT,
        )
    except CalledProcessError as cper:
        if error_message is None:
            error_message = f"Error during call to {args[0]}"
        raise InternalError(
            f"{error_message}:\n\t"
            + "\t".join(cper.output.decode("utf-8").splitlines(True))
        ) from cper


class Module:
    """
    Abstracts a SPIR-V binary module that is created by calling
    ``llvm-spirv`` translator on a LLVM IR binary module.
    """

    @staticmethod
    def _llvm_spirv():
        """Return path to llvm-spirv executable."""

        return dls.get_llvm_spirv_path()

    def __init__(self, context, llvmir, llvmbc):
        self._tmpdir = tempfile.mkdtemp()
        self._tempfiles = []
        self._finalized = False
        self.context = context
        self._llvmfile = None
        self._llvmir = llvmir
        self._llvmbc = llvmbc

    def __del__(self):
        # Remove all temporary files
        for afile in self._tempfiles:
            os.unlink(afile)
        # Remove directory
        os.rmdir(self._tmpdir)

    def _track_temp_file(self, name):
        path = os.path.join(self._tmpdir, f"{len(self._tempfiles)}-{name}")
        self._tempfiles.append(path)
        return path

    def _generate_spirv(self, llvm_spirv_args, ipath, opath):
        """
        Generate a spirv module from llvm bitcode.

        We use llvm-spirv tool to translate a llvm bitcode into spirv.

        Args:
            llvm_spirv_args: Args to be provided to llvm-spirv tool.
            ipath: Input file path of the llvm bitcode.
            opath: Output file path of the generated spirv.
        """
        llvm_spirv_flags = []
        if config.DEBUG:
            llvm_spirv_flags.append("--spirv-debug-info-version=ocl-100")

        llvm_spirv_tool = self._llvm_spirv()

        if config.DEBUG:
            print(f"Use llvm-spirv: {llvm_spirv_tool}")

        run_cmd(
            [llvm_spirv_tool, *llvm_spirv_args, "-o", opath, ipath],
            error_message="Error during lowering LLVM IR to SPIRV",
        )

    def load_llvm(self):
        """
        Load LLVM with "SPIR-V friendly" SPIR 2.0 spec
        """
        # Create temp file to store the input file
        llvm_path = self._track_temp_file("llvm-friendly-spir")
        with open(llvm_path, mode="wb") as tmp_llvm_ir:
            tmp_llvm_ir.write(self._llvmbc)

        self._llvmfile = llvm_path

    def finalize(self):
        """
        Finalize module and return the SPIR-V code
        """
        assert not self._finalized, "Module finalized already"

        # Generate SPIR-V from "friendly" LLVM-based SPIR 2.0
        spirv_path = self._track_temp_file("generated-spirv")

        # TODO: find better approach to set SPIRV compiler arguments. Workaround
        #  against caching intrinsic that sets this argument.
        # https://github.com/IntelPython/numba-dpex/issues/1262
        llvm_spirv_args = [
            "--spirv-ext=+SPV_EXT_shader_atomic_float_add",
            "--spirv-ext=+SPV_EXT_shader_atomic_float_min_max",
            "--spirv-ext=+SPV_INTEL_arbitrary_precision_integers",
            "--spirv-ext=+SPV_INTEL_variable_length_array",
        ]
        for key in list(self.context.extra_compile_options.keys()):
            if key == LLVM_SPIRV_ARGS:
                llvm_spirv_args = self.context.extra_compile_options[key]
            del self.context.extra_compile_options[key]

        if config.SAVE_IR_FILES != 0:
            # Dump the llvmir and llvmbc in file
            with open("generated_llvm.ir", "w", encoding="utf-8") as f:
                f.write(self._llvmir)
            with open("generated_llvm.bc", "wb") as f:
                f.write(self._llvmbc)

            print("Generated LLVM IR".center(80, "-"))
            print("generated_llvm.ir")
            print("".center(80, "="))
            print("Generated LLVM Bitcode".center(80, "-"))
            print("generated_llvm.bc")
            print("".center(80, "="))

        self._generate_spirv(
            llvm_spirv_args=llvm_spirv_args,
            ipath=self._llvmfile,
            opath=spirv_path,
        )

        if config.SAVE_IR_FILES != 0:
            # Dump the llvmir and llvmbc in file
            with open("generated_spirv.spir", "wb") as f1:
                with open(spirv_path, "rb") as f2:
                    spirv_content = f2.read()
                    f1.write(spirv_content)

            print("Generated SPIRV".center(80, "-"))
            print("generated_spirv.spir")
            print("".center(80, "="))

        # Read and return final SPIR-V
        with open(spirv_path, "rb") as fin:
            spirv = fin.read()

        self._finalized = True

        return spirv


def llvm_to_spirv(context, llvmir, llvmbc):
    """
    Generate SPIR-V from LLVM Bitcode.

    Args:
        context: Numba target context.
        llvmir: LLVM IR.
        llvmbc: LLVM Bitcode.

    Returns:
        spirv: SPIR-V binary.
    """
    mod = Module(context, llvmir, llvmbc)
    mod.load_llvm()
    return mod.finalize()
