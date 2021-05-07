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

# A wrapper to connect to the SPIR-V binaries (Tools, Translator).
# Currently, connect to commandline interface.
from __future__ import print_function, absolute_import
import sys
import os
from subprocess import check_call, CalledProcessError, call
import tempfile

from numba import config
from numba_dppy import config as dppy_config
from numba_dppy.target import LINK_ATOMIC, LLVM_SPIRV_ARGS


def _raise_bad_env_path(msg, path, extra=None):
    error_message = msg.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)


_real_check_call = check_call


def check_call(*args, **kwargs):
    # print("check_call:", *args, **kwargs)
    return _real_check_call(*args, **kwargs)


class CmdLine(object):
    def disassemble(self, ipath, opath):
        check_call(
            [
                "spirv-dis",
                # "--no-indent",
                # "--no-header",
                # "--raw-id",
                # "--offsets",
                "-o",
                opath,
                ipath,
            ]
        )

    def validate(self, ipath):
        check_call(["spirv-val", ipath])

    def optimize(self, ipath, opath):
        check_call(
            [
                "spirv-opt",
                # "--strip-debug",
                # "--freeze-spec-const",
                # "--eliminate-dead-const",
                # "--fold-spec-const-op-composite",
                # "--set-spec-const-default-value '<spec id>:<default value> ...'",
                # "--unify-const",
                # "--inline-entry-points-exhaustive",
                # "--flatten-decorations",
                # "--compact-ids",
                "-o",
                opath,
                ipath,
            ]
        )

    def generate(self, ipath, opath, llvm_spirv_args):
        # DRD : Temporary hack to get SPIR-V code generation to work.
        # The opt step is needed for:
        #     a) generate a bitcode file from the text IR file
        #     b) hoist all allocas to the enty block of the module
        # Get optimization level from NUMBA_OPT
        opt_level_option = f"-O{config.OPT}"

        check_call(["opt", opt_level_option, "-o", ipath + ".bc", ipath])

        if dppy_config.NATIVE_FP_ATOMICS == 1:
            llvm_spirv_root = dppy_config.LLVM_SPIRV_ROOT

            if llvm_spirv_root == "":
                # try to find ONEAPI root
                possible_oneapi_roots = ["/opt/intel/oneapi", "$A21_SDK_ROOT"]
                for path in possible_oneapi_roots:
                    path += "/compiler/latest/linux/bin"
                    path = os.path.expandvars(path)
                    if os.path.isfile(path + "/llvm-spirv"):
                        llvm_spirv_root = path
                        break

            if llvm_spirv_root == "":
                raise ValueError(
                    "Native floating point atomics require dpcpp provided llvm-spirv, "
                    "please specify the LLVM-SPIRV root directory using env variable "
                    "NUMBA_DPPY_LLVM_SPIRV_ROOT."
                )

            llvm_spirv_call_args = [path + "/llvm-spirv"]
        else:
            llvm_spirv_call_args = ["llvm-spirv"]
        if llvm_spirv_args is not None:
            llvm_spirv_call_args += llvm_spirv_args
        llvm_spirv_call_args += ["-o", opath, ipath + ".bc"]
        check_call(llvm_spirv_call_args)

        if dppy_config.SAVE_IR_FILES == 0:
            os.unlink(ipath + ".bc")

    def link(self, opath, binaries):
        params = ["spirv-link", "--allow-partial-linkage", "-o", opath]
        params.extend(binaries)

        check_call(params)


class Module(object):
    def __init__(self, context):
        """
        Setup
        """
        self._tmpdir = tempfile.mkdtemp()
        self._tempfiles = []
        self._cmd = CmdLine()
        self._finalized = False
        self.context = context

    def __del__(self):
        # Remove all temporary files
        for afile in self._tempfiles:
            if dppy_config.SAVE_IR_FILES != 0:
                print(afile)
            else:
                os.unlink(afile)
        # Remove directory
        if dppy_config.SAVE_IR_FILES == 0:
            os.rmdir(self._tmpdir)

    def _create_temp_file(self, name, mode="wb"):
        path = self._track_temp_file(name)
        fobj = open(path, mode=mode)
        return fobj, path

    def _track_temp_file(self, name):
        path = os.path.join(self._tmpdir, "{0}-{1}".format(len(self._tempfiles), name))
        self._tempfiles.append(path)
        return path

    def load_llvm(self, llvmir):
        """
        Load LLVM with "SPIR-V friendly" SPIR 2.0 spec
        """
        # Create temp file to store the input file
        tmp_llvm_ir, llvm_path = self._create_temp_file("llvm-friendly-spir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(llvmir.encode())

        self._llvmfile = llvm_path

    def finalize(self):
        """
        Finalize module and return the SPIR-V code
        """
        assert not self._finalized, "Module finalized already"

        # Generate SPIR-V from "friendly" LLVM-based SPIR 2.0
        spirv_path = self._track_temp_file("generated-spirv")

        binary_paths = [spirv_path]
        llvm_spirv_args = None
        for key in list(self.context.extra_compile_options.keys()):
            if key == LINK_ATOMIC:
                from .ocl.atomics import get_atomic_spirv_path

                binary_paths.append(get_atomic_spirv_path())
            if key == LLVM_SPIRV_ARGS:
                llvm_spirv_args = self.context.extra_compile_options[key]
            del self.context.extra_compile_options[key]

        self._cmd.generate(
            ipath=self._llvmfile, opath=spirv_path, llvm_spirv_args=llvm_spirv_args
        )

        if len(binary_paths) > 1:
            spirv_path = self._track_temp_file("linked-spirv")
            self._cmd.link(spirv_path, binary_paths)

        # Validate the SPIR-V code
        if dppy_config.SPIRV_VAL == 1:
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


# Public llvm_to_spirv function ###############################################


def llvm_to_spirv(context, bitcode):
    mod = Module(context)
    mod.load_llvm(bitcode)
    return mod.finalize()
