# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""A wrapper to connect to the SPIR-V binaries (Tools, Translator)."""

import os
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output

from numba_dpex import config
from numba_dpex.core.exceptions import InternalError
from numba_dpex.core.targets.kernel_target import LLVM_SPIRV_ARGS


def run_cmd(args, error_message=None):
    try:
        check_output(
            args,
            stderr=STDOUT,
        )
    except CalledProcessError as err:
        if error_message is None:
            error_message = f"Error during call to {args[0]}"
        raise InternalError(
            f"{error_message}:\n\t"
            + "\t".join(err.output.decode("utf-8").splitlines(True))
        )


class CmdLine:
    def disassemble(self, ipath, opath):
        """
        Disassemble a spirv module.

        Args:
            ipath: Input file path of the spirv module.
            opath: Output file path of the disassembled spirv module.
        """
        flags = []
        run_cmd(
            ["spirv-dis", *flags, "-o", opath, ipath],
            error_message="Error during SPIRV disassemble",
        )

    def validate(self, ipath):
        """
        Validate a spirv module.

        Args:
            ipath: Input file path of the spirv module.
        """
        flags = []
        run_cmd(
            ["spirv-val", *flags, ipath],
            error_message="Error during SPIRV validation",
        )

    def optimize(self, ipath, opath):
        """
        Optimize a spirv module.

        Args:
            ipath: Input file path of the spirv module.
            opath: Output file path of the optimized spirv module.
        """
        flags = []
        run_cmd(
            ["spirv-opt", *flags, "-o", opath, ipath],
            error_message="Error during SPIRV optimization",
        )

    def generate(self, llvm_spirv_args, ipath, opath):
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

    @staticmethod
    def _llvm_spirv():
        """Return path to llvm-spirv executable."""

        try:
            import dpcpp_llvm_spirv as dls
        except ImportError:
            raise ImportError("Cannot import dpcpp-llvm-spirv package")

        result = dls.get_llvm_spirv_path()
        return result


class Module(object):
    def __init__(self, context, llvmir, llvmbc):
        """
        Setup
        """
        self._tmpdir = tempfile.mkdtemp()
        self._tempfiles = []
        self._cmd = CmdLine()
        self._finalized = False
        self.context = context

        self._llvmir = llvmir
        self._llvmbc = llvmbc

    def __del__(self):
        # Remove all temporary files
        for afile in self._tempfiles:
            os.unlink(afile)
        # Remove directory
        os.rmdir(self._tmpdir)

    def _track_temp_file(self, name):
        path = os.path.join(
            self._tmpdir, "{0}-{1}".format(len(self._tempfiles), name)
        )
        self._tempfiles.append(path)
        return path

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
        llvm_spirv_args = ["--spirv-ext=+SPV_EXT_shader_atomic_float_add"]
        for key in list(self.context.extra_compile_options.keys()):
            if key == LLVM_SPIRV_ARGS:
                llvm_spirv_args = self.context.extra_compile_options[key]
            del self.context.extra_compile_options[key]

        if config.SAVE_IR_FILES != 0:
            # Dump the llvmir and llvmbc in file
            with open("generated_llvm.ir", "w") as f:
                f.write(self._llvmir)
            with open("generated_llvm.bc", "wb") as f:
                f.write(self._llvmbc)

            print("Generated LLVM IR".center(80, "-"))
            print("generated_llvm.ir")
            print("".center(80, "="))
            print("Generated LLVM Bitcode".center(80, "-"))
            print("generated_llvm.bc")
            print("".center(80, "="))

        self._cmd.generate(
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

        # Validate the SPIR-V code
        if config.SPIRV_VAL == 1:
            try:
                self._cmd.validate(ipath=spirv_path)
            except CalledProcessError:
                print("SPIR-V Validation failed...")
                pass
            else:
                # Optimize SPIR-V code
                opt_path = self._track_temp_file("optimized-spirv")
                self._cmd.optimize(ipath=spirv_path, opath=opt_path)

                if config.DUMP_ASSEMBLY:
                    # Disassemble optimized SPIR-V code
                    dis_path = self._track_temp_file("disassembled-spirv")
                    self._cmd.disassemble(ipath=opt_path, opath=dis_path)
                    with open(dis_path, "rb") as fin_opt:
                        print("ASSEMBLY".center(80, "-"))
                        print(fin_opt.read())
                        print("".center(80, "="))

        # Read and return final SPIR-V (not optimized!)
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
