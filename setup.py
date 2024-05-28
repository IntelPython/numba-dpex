# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import os
import re

import versioneer
from setuptools import find_namespace_packages, find_packages
from skbuild import setup

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

# Keep same style as versioneer in case ve would want to switch back
with open(os.path.join("numba_dpex", "_version.py"), "w") as fh:
    print(
        "# SPDX-FileCopyrightText: 2024 Intel Corporation\n#\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n\ndef get_versions():\n"
        '    return {"version": "' + __version__ + '"}',
        file=fh,
    )


# Main setup
setup(
    version=__version__,
    # https://github.com/pypa/pip/issues/11221
    url="https://github.com/IntelPython/numba-dpex",
    # Must be passed vis setup.py:
    # https://github.com/scikit-build/scikit-build/issues/864
    # TODO: switch to pyproject toml after switching to scikit-build-core
    packages=find_packages(".")
    + find_namespace_packages(".", include=["numba_dpex.examples.*"]),
    # Needs for examples.
    # TODO: change to false once move examples out of package.
    include_package_data=True,
    cmake_args=[
        "-DNUMBA_DPEX_VERSION:STRING={0:s}".format(
            to_cmake_format(str(__version__))
        ),
    ],
)
