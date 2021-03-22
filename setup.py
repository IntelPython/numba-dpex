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

import os
import sys
import setuptools.command.install as orig_install
import setuptools.command.develop as orig_develop
import subprocess
import shutil
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

import versioneer
import sys


IS_WIN = False
IS_LIN = False

if "linux" in sys.platform:
    IS_LIN = True
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


def get_ext_modules():
    ext_modules = []

    import numba

    ext_dppy = Extension(
        name="numba_dppy._dppy_rt",
        sources=["numba_dppy/dppy_rt.c"],
        include_dirs=[numba.core.extending.include_path()],
    )
    ext_modules += [ext_dppy]

    dpnp_present = False
    try:
        import dpnp
    except:
        pass
    else:
        dpnp_present = True

    if dpnp_present:
        dpnp_lib_path = []
        dpnp_lib_path += [os.path.dirname(dpnp.__file__)]
        ext_dpnp_glue = Extension(
            name="numba_dppy.dpnp_glue.dpnp_fptr_interface",
            sources=["numba_dppy/dpnp_glue/dpnp_fptr_interface.pyx"],
            include_dirs=[dpnp.get_include()],
            libraries=["dpnp_backend_c"],
            library_dirs=dpnp_lib_path,
            runtime_library_dirs=dpnp_lib_path,
            language="c++",
        )
        ext_modules += [ext_dpnp_glue]

    if dpnp_present:
        return cythonize(ext_modules)
    else:
        return ext_modules


class install(orig_install.install):
    def run(self):
        spirv_compile()
        super().run()


class develop(orig_develop.develop):
    def run(self):
        spirv_compile()
        super().run()


def _get_cmdclass():
    cmdclass = versioneer.get_cmdclass()
    cmdclass["install"] = install
    cmdclass["develop"] = develop
    return cmdclass


def spirv_compile():
    if IS_LIN:
        compiler = os.path.join(
            os.environ.get("ONEAPI_ROOT"), "compiler/latest/linux", "bin/clang"
        )
    if IS_WIN:
        compiler = os.path.join(
            os.environ.get("ONEAPI_ROOT"), "compiler/latest/windows", "bin/clang.exe"
        )
    clang_args = [
        compiler,
        "-flto",
        "-target",
        "spir64-unknown-unknown",
        "-c",
        "-x",
        "cl",
        "-emit-llvm",
        "-cl-std=CL2.0",
        "-Xclang",
        "-finclude-default-header",
        "numba_dppy/ocl/atomics/atomic_ops.cl",
        "-o",
        "numba_dppy/ocl/atomics/atomic_ops.bc",
    ]
    spirv_args = [
        "llvm-spirv",
        "-o",
        "numba_dppy/ocl/atomics/atomic_ops.spir",
        "numba_dppy/ocl/atomics/atomic_ops.bc",
    ]
    if IS_LIN:
        subprocess.check_call(clang_args, stderr=subprocess.STDOUT, shell=False)
        subprocess.check_call(spirv_args, stderr=subprocess.STDOUT, shell=False)
    if IS_WIN:
        subprocess.check_call(clang_args, stderr=subprocess.STDOUT, shell=True)
        subprocess.check_call(spirv_args, stderr=subprocess.STDOUT, shell=True)


packages = find_packages(include=["numba_dppy", "numba_dppy.*"])
build_requires = ["cython"]
install_requires = [
    "numba",
    "dpctl",
]

metadata = dict(
    name="numba-dppy",
    version=versioneer.get_version().split("+")[0],
    cmdclass=_get_cmdclass(),
    description="Numba extension for Intel CPU and GPU backend",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IntelPython/numba-dppy",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
    include_package_data=True,
    ext_modules=get_ext_modules(),
    author="Intel Corporation",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Compilers",
    ],
    entry_points={
        "numba_extensions": [
            "init = numba_dppy.numpy_usm_shared:numba_register",
        ]
    },
)

setup(**metadata)
