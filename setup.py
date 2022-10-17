# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

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

    import dpctl
    import numba

    dpctl_runtime_library_dirs = []

    if IS_LIN:
        dpctl_runtime_library_dirs.append(os.path.dirname(dpctl.__file__))

    ext_usm_alloc = Extension(
        name="numba_dpex._usm_allocators_ext",
        sources=["numba_dpex/dpctl_iface/usm_allocators_ext.c"],
        include_dirs=[numba.core.extending.include_path(), dpctl.get_include()],
        libraries=["DPCTLSyclInterface"],
        library_dirs=[os.path.dirname(dpctl.__file__)],
        runtime_library_dirs=dpctl_runtime_library_dirs,
    )
    ext_modules += [ext_usm_alloc]

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
        "-o",
        "numba_dpex/ocl/atomics/atomic_ops.spir",
        "numba_dpex/ocl/atomics/atomic_ops.bc",
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
    import shutil

    result = None

    if result is None:
        # use llvm-spirv from dpcpp package.
        # assume dpcpp from .../bin folder.
        # assume llvm-spirv from .../bin-llvm folder.
        dpcpp_path = shutil.which("dpcpp")
        if dpcpp_path is not None:
            bin_llvm = os.path.dirname(dpcpp_path) + "/../bin-llvm/"
            bin_llvm = os.path.normpath(bin_llvm)
            result = shutil.which("llvm-spirv", path=bin_llvm)

    if result is None:
        result = "llvm-spirv"

    return result


packages = find_packages(include=["numba_dpex", "numba_dpex.*"])
build_requires = ["cython"]
install_requires = [
    "numba >={},<{}".format("0.54.0", "0.56"),
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
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Compilers",
    ],
    entry_points={
        "numba_extensions": [
            "init = numba_dpex.numpy_usm_shared:numba_register",
        ]
    },
)

setup(**metadata)
