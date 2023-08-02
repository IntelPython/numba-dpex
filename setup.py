# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import sysconfig

import dpctl
import numba
import numpy
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
    except ImportError:
        raise ImportError("dpnp should be installed to build numba-dpex")

    dpctl_runtime_library_dirs = []

    if IS_LIN:
        dpctl_runtime_library_dirs.append(os.path.dirname(dpctl.__file__))

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

    return ext_modules


packages = find_packages(
    include=["numba_dpex", "numba_dpex.*", "_dpexrt_python"]
)
install_requires = [
    "numba >={}".format("0.57"),
    "dpctl",
    "packaging",
]

metadata = dict(
    name="numba-dpex",
    version=versioneer.get_version(),
    description="An extension for Numba to add data-parallel offload capability",
    url="https://github.com/IntelPython/numba-dpex",
    packages=packages,
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Compilers",
    ],
    entry_points={},
)

setup(**metadata)
