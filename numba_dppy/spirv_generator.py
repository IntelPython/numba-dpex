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

"""A wrapper to connect to the SPIR-V binaries (Tools, Translator)."""

import sys
import os
from subprocess import check_call, CalledProcessError, call
import tempfile

from numba_dppy import config
from numba_dppy.target import LINK_ATOMIC, LLVM_SPIRV_ARGS


def _raise_bad_env_path(msg, path, extra=None):
    error_message = msg.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)


_real_check_call = check_call


def check_call(*args, **kwargs):
    return _real_check_call(*args, **kwargs)


class CmdLine:
    def disassemble(self, ipath, opath):
        """
        Disassemble a spirv module.

        Args:
            ipath: Input file path of the spirv module.
            opath: Output file path of the disassembled spirv module.
        """
        flags = []
        check_call(["spirv-dis", *flags, "-o", opath, ipath])

    def validate(self, ipath):
        """
        Validate a spirv module.

        Args:
            ipath: Input file path of the spirv module.
        """
        flags = []
        check_call(["spirv-val", *flags, ipath])

    def optimize(self, ipath, opath):
        """
        Optimize a spirv module.

        Args:
            ipath: Input file path of the spirv module.
            opath: Output file path of the optimized spirv module.
        """
        flags = []
        check_call(["spirv-opt", *flags, "-o", opath, ipath])

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

        llvm_spirv_tool = "llvm-spirv"
        if config.NATIVE_FP_ATOMICS == 1:
            llvm_spirv_root = config.LLVM_SPIRV_ROOT

            if llvm_spirv_root == "":
                raise ValueError(
                    "Native floating point atomics require dpcpp provided "
                    "llvm-spirv, please specify the LLVM-SPIRV root directory "
                    "using env variable NUMBA_DPPY_LLVM_SPIRV_ROOT."
                )

            llvm_spirv_tool = llvm_spirv_root + "/llvm-spirv"

        check_call([llvm_spirv_tool, *llvm_spirv_args, "-o", opath, ipath])

    def link(self, opath, binaries):
        """
        Link spirv modules.

        Args:
            opath: Output file path of the linked final spirv.
            binaries: Spirv modules to be linked.
        """
        flags = ["--allow-partial-linkage"]
        check_call(["spirv-link", *flags, "-o", opath, *binaries])


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

    def _create_temp_file(self, name, mode="wb"):
        path = self._track_temp_file(name)
        fobj = open(path, mode=mode)
        return fobj, path

    def _track_temp_file(self, name):
        path = os.path.join(self._tmpdir, "{0}-{1}".format(len(self._tempfiles), name))
        self._tempfiles.append(path)
        return path

    def load_llvm(self):
        """
        Load LLVM with "SPIR-V friendly" SPIR 2.0 spec
        """
        # Create temp file to store the input file
        tmp_llvm_ir, llvm_path = self._create_temp_file("llvm-friendly-spir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(self._llvmbc)

        self._llvmfile = llvm_path

    def finalize(self):
        """
        Finalize module and return the SPIR-V code
        """
        assert not self._finalized, "Module finalized already"

        # Generate SPIR-V from "friendly" LLVM-based SPIR 2.0
        spirv_path = self._track_temp_file("generated-spirv")

        binary_paths = [spirv_path]

        llvm_spirv_args = []
        for key in list(self.context.extra_compile_options.keys()):
            if key == LINK_ATOMIC:
                from .ocl.atomics import get_atomic_spirv_path

                binary_paths.append(get_atomic_spirv_path())
            elif key == LLVM_SPIRV_ARGS:
                llvm_spirv_args = self.context.extra_compile_options[key]
            del self.context.extra_compile_options[key]

        self._cmd.generate(
            llvm_spirv_args=llvm_spirv_args, ipath=self._llvmfile, opath=spirv_path
        )

        if len(binary_paths) > 1:
            spirv_path = self._track_temp_file("linked-spirv")
            self._cmd.link(spirv_path, binary_paths)

        if config.SAVE_IR_FILES != 0:
            # Dump the llvmir and llvmbc in file
            with open("generated_llvm.ir", "w") as f:
                f.write(self._llvmir)
            with open("generated_llvm.bc", "wb") as f:
                f.write(self._llvmbc)
            with open("generated_spirv.spir", "wb") as f1:
                with open(spirv_path, "rb") as f2:
                    spirv_content = f2.read()
                    f1.write(spirv_content)

            print("Generated LLVM IR".center(80, "-"))
            print("generated_llvm.ir")
            print("".center(80, "="))
            print("Generated LLVM Bitcode".center(80, "-"))
            print("generated_llvm.bc")
            print("".center(80, "="))
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
