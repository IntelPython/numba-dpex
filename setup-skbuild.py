import importlib.machinery as imm
import os
import shutil
import sys
import sysconfig

import dpctl
import numba
import numpy
from setuptools import Extension, find_packages
from skbuild import setup

import versioneer


def to_cmake_format(version):
    version = version.strip()
    parts = version.split("+")
    tag, dist = parts[0], parts[1].split(".")[0]
    return tag + "." + dist


"""
Test if system is WIN
"""
is_windows = sys.platform.startswith("win") or sys.platform.startswith("cyg")


"""
Set compiler
"""
cc = "icx.exe" if is_windows else "icx"
cxx = "icpx.exe" if is_windows else "icpx"

icx = shutil.which(cc)
if icx:
    os.environ["CC"] = cc

icpx = shutil.which(cxx)
if icpx:
    os.environ["CXX"] = cxx


"""
Get the project version
"""
__version__ = versioneer.get_version()
os.environ["NUMBA_DPEX_VERSION"] = to_cmake_format(str(__version__))


"""
Set project auxilary data like readme and licence files
"""
with open("README.md") as f:
    __readme_file__ = f.read()


def get_ext_modules():
    ext_modules = []

    try:
        import dpnp
    except ImportError:
        raise ImportError("dpnp should be installed to build numba-dpex")

    dpctl_runtime_library_dirs = []

    if not is_windows:
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


setup(
    name="numba-dpex",
    version=__version__,
    description="An extension for Numba to add data-parallel offload capability",
    long_description=__readme_file__,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
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
    keywords="sycl python3 numba numpy intel mkl oneapi gpu dpcpp",
    platforms=["Linux", "Windows"],
    author="Intel Corporation",
    url="https://github.com/IntelPython/numba-dpex",
    install_requires=["numba >={}".format("0.57"), "dpctl", "packaging"],
    packages=find_packages(["numba_dpex", "numba_dpex.*"]),
    # package_data={
    #     "dpnp": [
    #         "libdpnp_backend_c.so",
    #         "dpnp_backend_c.lib",
    #         "dpnp_backend_c.dll",
    #     ]
    # },
    include_package_data=True,
    zip_safe=False,
)
