# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import sysconfig

import dpctl
import numba
import numpy
import setuptools.command.develop as orig_develop
import setuptools.command.install as orig_install
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import versioneer

IS_WIN = False
IS_LIN = False

if "linux" in sys.platform:
    IS_LIN = True
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True


def get_ext_modules():
    ext_modules = []

    try:
        import dpnp

        dpnp_present = True
    except ImportError:
        if int(os.environ.get("NUMBA_DPEX_BUILD_SKIP_NO_DPNP", 0)):
            dpnp_present = False
        else:
            raise ImportError("DPNP is not available")

    dpctl_runtime_library_dirs = []

    if IS_LIN:
        dpctl_runtime_library_dirs.append(os.path.dirname(dpctl.__file__))

    if dpnp_present:
        dpnp_lib_path = []
        dpnp_lib_path += [os.path.dirname(dpnp.__file__)]
        ext_dpnp_iface = Extension(
            name="numba_dpex.dpnp_iface.dpnp_fptr_interface",
            sources=["numba_dpex/dpnp_iface/dpnp_fptr_interface.pyx"],
            include_dirs=[dpnp.get_include(), dpctl.get_include()],
            libraries=["dpnp_backend_c"],
            library_dirs=dpnp_lib_path,
            runtime_library_dirs=(dpnp_lib_path if IS_LIN else []),
            language="c++",
        )
        ext_modules += [ext_dpnp_iface]

    ext_dpexrt_python = Extension(
        name="numba_dpex.core.runtime._dpexrt_python",
        sources=[
            "numba_dpex/core/runtime/_dpexrt_python.c",
            "numba_dpex/core/runtime/_nrt_helper.c",
            "numba_dpex/core/runtime/_nrt_python_helper.c",
        ],
        libraries=["DPCTLSyclInterface"],
        library_dirs=[os.path.dirname(dpctl.__file__)],
        runtime_library_dirs=dpctl_runtime_library_dirs,
        include_dirs=[
            sysconfig.get_paths()["include"],
            numba.extending.include_path(),
            numpy.get_include(),
            dpctl.get_include(),
        ],
    )

    ext_modules += [ext_dpexrt_python]

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
        compiler = "icx"
    if IS_WIN:
        compiler = "clang.exe"

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
        "numba_dpex/ocl/atomics/atomic_ops.cl",
        "-o",
        "numba_dpex/ocl/atomics/atomic_ops.bc",
    ]
    spirv_args = [
        _llvm_spirv(),
        "--spirv-max-version",
        "1.1",
        "numba_dpex/ocl/atomics/atomic_ops.bc",
        "-o",
        "numba_dpex/ocl/atomics/atomic_ops.spir",
    ]
    subprocess.check_call(
        clang_args,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        shell=False,
    )
    subprocess.check_call(
        spirv_args,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        shell=False,
    )


def _llvm_spirv():
    """Return path to llvm-spirv executable."""

    try:
        import dpcpp_llvm_spirv as dls
    except ImportError:
        raise ImportError("Cannot import dpcpp-llvm-spirv package")

    result = dls.get_llvm_spirv_path()

    return result


packages = find_packages(
    include=["numba_dpex", "numba_dpex.*", "_dpexrt_python"]
)
build_requires = ["cython"]
install_requires = [
    "numba >={}".format("0.56"),
    "dpctl",
    "packaging",
]

metadata = dict(
    name="numba-dpex",
    version=versioneer.get_version(),
    cmdclass=_get_cmdclass(),
    description="An extension for Numba to add data-parallel offload capability",
    url="https://github.com/IntelPython/numba-dpex",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
    ext_modules=get_ext_modules(),
    author="Intel Corporation",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Compilers",
    ],
    entry_points={},
)

setup(**metadata)
