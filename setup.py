# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import re

from setuptools import find_packages
from skbuild import setup

import versioneer

"""Top level setup.py file. Uses scikit-build.

    This will build the numba_dpex project. There are two ways to run this file.
    First command line argument is `install` and the other is `develop`. The
    argument `install` will build the _dpexrt_runtime target and install all the
    python modules into _skbuild build directory, where `develop` will just
    build the _dpexrt_runtime target and will not copy any python module into
    _skbuild.

    `install` command:
        ~$ python setup.py install

    `develop` command:
        ~$ python setup.py develop

    To uninstall:
        ~$ pip uninstall numba-dpex

    NOTE: This script doesn't support pypa/build, pypa/installer or other
    standards-based tools like pip, yet.

    TODO: Support `pip install`
"""


def to_cmake_format(version: str):
    """Convert pep440 version string into a cmake compatible string."""
    # cmake version just support up to 4 numbers separated by dot.
    # https://peps.python.org/pep-0440/
    # https://cmake.org/cmake/help/latest/command/project.html

    match = re.search(r"^\d+(?:\.\d+(?:\.\d+(?:\.\d+)?)?)?", version)
    if not match:
        raise Exception("Unsupported version")

    return match.group(0)


# Get the project version
__version__ = versioneer.get_version()


# Set project auxilary data like readme and licence files
with open("README.md", "r") as f:
    __readme__ = "".join(line for line in f.readlines()[12:35])


# Main setup
setup(
    name="numba-dpex",
    version=__version__,
    description="An extension for Numba to add data-parallel offload capability",
    long_description=__readme__,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Compilers",
    ],
    keywords="sycl python3 numba numpy intel mkl oneapi gpu dpcpp",
    platforms=["Linux", "Windows"],
    author="Intel Corporation",
    url="https://github.com/IntelPython/numba-dpex",
    install_requires=["numba >={0:s}".format("0.58"), "dpctl", "packaging"],
    packages=find_packages("."),
    include_package_data=True,
    zip_safe=False,
    cmake_args=[
        "-DNUMBA_DPEX_VERSION:STRING={0:s}".format(
            to_cmake_format(str(__version__))
        ),
        "-DIS_INSTALL:BOOL={0:s}".format("TRUE" if is_install else "FALSE"),
        "-DIS_DEVELOP:BOOL={0:s}".format("TRUE" if is_develop else "FALSE"),
    ],
)
