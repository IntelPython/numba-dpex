# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig

import dpctl
import numba
import numpy
import setuptools.command.develop as orig_develop
import setuptools.command.install as orig_install
from setuptools import Extension, find_packages  # , setup
from skbuild import setup

import versioneer

IS_LINUX = False
IS_WINDOWS = False
IS_OSX = False


if "linux" in sys.platform:
    IS_LINUX = True
elif "win32" in sys.platform or "cygwin" in sys.platform:
    IS_WINDOWS = True
elif "darwin" in sys.platform:
    IS_OSX = True
else:
    raise RuntimeError("Unknown platform.")


def get_spriv_compiler():
    if IS_LINUX:
        return "icx"
    elif IS_WINDOWS:
        return "clang.exe"
    elif IS_OSX:
        return "clang"
    else:
        return "gcc"


def ensure_dpnp():
    try:
        import dpnp
    except ImportError:
        raise ImportError("dpnp should be installed to build numba-dpex")


def get_conda_prefix():
    conda_prefix = None
    if "CONDA_PREFIX" in os.environ:
        conda_prefix = os.environ["CONDA_PREFIX"]
        if os.path.exists(conda_prefix):
            return conda_prefix
    else:
        conda_prefix = sys.exec_prefix
        if os.path.exists(conda_prefix) and "conda" in conda_prefix:
            return conda_prefix
        else:
            raise RuntimeError("Conda prefix not found or unusable.")


def find_sycl_header_path():
    header_path, lib_path = None, None
    # conda_prefix = get_conda_prefix()
    # if conda_prefix:
    #     header_path = os.path.join(conda_prefix, "include")
    #     lib_path = os.path.join(conda_prefix, "lib")
    if "CMPLR_ROOT" in os.environ:
        cmplr_root = os.environ["CMPLR_ROOT"]
        lib_path = os.path.join(cmplr_root, "linux/lib")
        header_path = os.path.join(cmplr_root, "linux/include")
    elif "LD_LIBRARY_PATH" in os.environ:
        ldlib_path = os.environ["LD_LIBRARY_PATH"]
        paths = ldlib_path.split(":")
        lib_path = None
        for path in paths:
            if "compiler" in path and "lib" in path:
                lib_path = path
                break
        if lib_path is not None:
            head = lib_path.split("lib")[0]
            header_path = os.path.join(head, "include")
        else:
            raise RuntimeError("SYCL lib path not found in LD_LIBRARY_PATH.")
    else:
        raise RuntimeError(
            "Neither of CONDA_PREFIX, LD_LIBRARY_PATH nor CMPLR_ROOT is configured."
        )
    if header_path and lib_path:
        if os.path.exists(header_path) and os.path.exists(lib_path):
            return header_path, lib_path
        else:
            raise FileNotFoundError(
                "Header ('{0:s}') and lib ('{1:s}') don't exist.".format(
                    header_path, lib_path
                )
            )
    else:
        raise RuntimeError(
            "Header path ({0:s}) and library path ({1:s}) don't exist.".format(
                header_path, lib_path
            )
        )


def find_mkl_header_path():
    header_path, lib_path = None, None
    # conda_prefix = get_conda_prefix()
    # if conda_prefix:
    #     header_path = os.path.join(conda_prefix, "include")
    #     lib_path = os.path.join(conda_prefix, "lib")
    if "MKLROOT" in os.environ:
        mklroot = os.environ["MKLROOT"]
        lib_path = os.path.join(mklroot, "lib/intel64")
        header_path = os.path.join(mklroot, "include")
    elif "LD_LIBRARY_PATH" in os.environ:
        ldlib_path = os.environ["LD_LIBRARY_PATH"]
        paths = ldlib_path.split(":")
        lib_path = None
        for path in paths:
            if "mkl" in path and "lib" in path:
                lib_path = path
                break
        if lib_path is not None:
            head = lib_path.split("lib")[0]
            header_path = os.path.join(head, "include")
        else:
            raise RuntimeError("MKL lib path not found in LD_LIBRARY_PATH.")
    else:
        raise RuntimeError(
            "Neither of CONDA_PREFIX, LD_LIBRARY_PATH nor CMPLR_ROOT is configured."
        )
    if header_path and lib_path:
        if os.path.exists(header_path) and os.path.exists(lib_path):
            return header_path, lib_path
        else:
            raise FileNotFoundError(
                "Header ('{0:s}') and lib ('{1:s}') don't exist.".format(
                    header_path, lib_path
                )
            )
    else:
        raise RuntimeError(
            "Header path ({0:s}) and library path ({1:s}) don't exist.".format(
                header_path, lib_path
            )
        )


def get_ext_modules():
    library_dirs = []
    runtime_library_dirs = []

    sycl_header, sycl_lib = find_sycl_header_path()
    mkl_header, mkl_lib = find_mkl_header_path()

    library_dirs.append(os.path.dirname(dpctl.__file__))

    if IS_LINUX:
        runtime_library_dirs.append(os.path.dirname(dpctl.__file__))

    ext_dpexrt_python = Extension(
        name="numba_dpex.core.runtime._dpexrt_python",
        sources=[
            "numba_dpex/core/runtime/_dpexrt_python.c",
            "numba_dpex/core/runtime/_nrt_helper.c",
            "numba_dpex/core/runtime/_nrt_python_helper.c",
        ],
        libraries=["DPCTLSyclInterface"],
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        include_dirs=[
            sysconfig.get_paths()["include"],
            numba.extending.include_path(),
            numpy.get_include(),
            dpctl.get_include(),
        ],
    )

    library_dirs.append(mkl_lib)
    ext_onemkl_lapack = Extension(  # noqa: F841
        name="numba_dpex.onemkl._dpex_lapack_iface",
        sources=["numba_dpex/onemkl/_dpex_lapack_iface.cpp"],
        libraries=[
            "mkl_sycl",
            "mkl_intel_ilp64",
            "mkl_tbb_thread",
            "mkl_core",
            "sycl",
            "OpenCL",
            "pthread",
            "m",
            "dl",
        ],
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        include_dirs=[
            sysconfig.get_paths()["include"],
            numba.extending.include_path(),
            numpy.get_include(),
            dpctl.get_include(),
            sycl_header,
            os.path.join(sycl_header, "sycl"),
            mkl_header,
        ],
    )

    ext_modules = [ext_dpexrt_python]  # , ext_onemkl_lapack]

    return ext_modules


# icx -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2
#     -Wall -Wformat -Wformat-security -fstack-protector-all
#     -D_FORTIFY_SOURCE=2 -fpic -fPIC -O2
#     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
#     -Wformat -Wformat-security -fstack-protector-all -D_FORTIFY_SOURCE=2 -fpic -fPIC -O2
#     -march=nocona -mtune=haswell -ftree-vectorize -fPIC
#     -fstack-protector-strong -fno-plt -O2 -ffunction-sections
#     -pipe -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
#     -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
#     -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
#     -fPIC -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include/python3.9
#     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages
#     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/numpy/core/include
#     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/dpctl/include
#     -I/localdisk/intel/oneapi/compiler/2023.1.0/linux/include
#     -I/localdisk/intel/oneapi/mkl/2023.1.0/include
#     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include/python3.9
#     -c numba_dpex/onemkl/_dpex_lapack_iface.cpp
#     -o build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o

# icpx -shared -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
#     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
#     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
#     -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro
#     -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections
#     -Wl,-rpath,/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
#     -Wl,-rpath-link,/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
#     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
#     -march=nocona -mtune=haswell -ftree-vectorize -fPIC
#     -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe
#     -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
#     -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
#     -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
#     build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o
#     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/dpctl
#     -L/localdisk/intel/oneapi/mkl/2023.1.0/lib/intel64
#     -Wl,--enable-new-dtags,-R/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/dpctl
#     -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread
#     -lm -ldl
#     -o build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so

# -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..
# -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..

# -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..
# -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..


def compile_onemkl_lapack_extension():
    conda_prefix = get_conda_prefix()
    sycl_header, sycl_lib = find_sycl_header_path()
    mkl_header, mkl_lib = find_mkl_header_path()

    print(conda_prefix)
    compile_cmd = """
    icx -Wno-unused-result -Wsign-compare -Wall -Wformat -Wformat-security
        -fwrapv -fstack-protector-all -ftree-vectorize -fPIC
        -fstack-protector-strong -fno-plt -ffunction-sections
        -pipe -march=nocona -mtune=haswell
        -isystem {0:s}/include
        -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
        -I{0:s}/include
        -I{0:s}/include/python3.9
        -I{0:s}/lib/python3.9/site-packages
        -I{0:s}/lib/python3.9/site-packages/numpy/core/include
        -I{0:s}/lib/python3.9/site-packages/dpctl/include
        -I{1:s}
        -I{2:s}
        -c numba_dpex/onemkl/_dpex_lapack_iface.cpp
        -o build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o
    """.format(
        conda_prefix, sycl_header, mkl_header
    )

    link_cmd = """
    icpx -shared -pipe
        -Wl,-O2
        -Wl,--sort-common
        -Wl,--as-needed
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,--disable-new-dtags
        -Wl,--gc-sections
        -Wl,-rpath,{0:s}/lib
        -Wl,-rpath-link,{0:s}/lib
        -Wl,--enable-new-dtags,-R{0:s}/lib/python3.9/site-packages/dpctl
        -march=nocona -mtune=haswell
        -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -ffunction-sections
        -isystem {0:s}/include
        -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
        build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o
        -L{0:s}/lib
        -L{0:s}/lib/python3.9/site-packages/dpctl
        -L{1:s}
        -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread
        -lm -ldl
        -o build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so
    """.format(
        conda_prefix, mkl_lib
    )

    print("compiling numba_dpex/onemkl/_dpex_lapack_iface.cpp ...")

    compile_cmd = re.sub(r"\n+", " ", re.sub(r"\s+", " ", compile_cmd)).strip()
    os.makedirs(
        "./build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl", exist_ok=True
    )

    fout = open("/tmp/compile.sh", "w")
    fout.write("#/bin/sh\n")
    fout.write(compile_cmd + "\n")
    # st = os.stat("/tmp/compile.sh")
    # os.chmod("/tmp/compile.sh", st.st_mode | stat.S_IEXEC)
    fout.close()

    try:
        output = subprocess.check_call(
            ["sh /tmp/compile.sh"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            shell=True,
        )
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
    else:
        print("Output: \n{}\n".format(output))

    print(
        "linking build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so ..."
    )

    link_cmd = re.sub(r"\n+", " ", re.sub(r"\s+", " ", link_cmd)).strip()
    os.makedirs(
        "./build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl", exist_ok=True
    )

    fout = open("/tmp/link.sh", "w")
    fout.write("#/bin/sh\n")
    fout.write(link_cmd + "\n")
    # st = os.stat("/tmp/link.sh")
    # os.chmod("/tmp/link.sh", st.st_mode | stat.S_IEXEC)
    fout.close()

    subprocess.check_call(
        ["sh /tmp/link.sh"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        shell=True,
    )

    print(
        "copying ./build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so -> ./numba_dpex/onemkl"
    )
    shutil.copy(
        "./build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so",
        "./numba_dpex/onemkl",
    )


def _llvm_spirv():
    """Return path to llvm-spirv executable."""

    try:
        import dpcpp_llvm_spirv as dls
    except ImportError:
        raise ImportError("Cannot import dpcpp-llvm-spirv package")

    result = dls.get_llvm_spirv_path()

    return result


def spirv_compile():
    clang_args = [
        get_spriv_compiler(),
        "-flto",
        "-fveclib=none",
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
        "1.0",
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


class install(orig_install.install):
    def run(self):
        spirv_compile()
        # compile_onemkl_lapack_extension()
        super().run()


class develop(orig_develop.develop):
    def run(self):
        spirv_compile()
        # compile_onemkl_lapack_extension()
        super().run()


def _get_cmdclass():
    cmdclass = versioneer.get_cmdclass()
    cmdclass["install"] = install
    cmdclass["develop"] = develop
    return cmdclass


packages = find_packages(
    include=[
        "numba_dpex",
        "numba_dpex.*",
        # "_dpexrt_python",
        # "_dpex_lapack_iface",
    ]
)

install_requires = [
    "numba >={}".format("0.57"),
    "dpctl",
    "packaging",
]

metadata = dict(
    name="numba-dpex",
    version=versioneer.get_version(),
    # cmdclass=_get_cmdclass(),
    description="An extension for Numba to add data-parallel offload capability",
    url="https://github.com/IntelPython/numba-dpex",
    packages=packages,
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
    # ext_modules=get_ext_modules(),
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


# def get_conda_prefix():
#     conda_prefix = None
#     if "CONDA_PREFIX" in os.environ:
#         conda_prefix = os.environ["CONDA_PREFIX"]
#         if os.path.exists(conda_prefix):
#             return conda_prefix
#     else:
#         conda_prefix = sys.exec_prefix
#         if os.path.exists(conda_prefix) and "conda" in conda_prefix:
#             return conda_prefix
#         else:
#             raise RuntimeError("Conda prefix not found or unusable.")


# def find_sycl_header_path():
#     header_path, lib_path = None, None
#     # conda_prefix = get_conda_prefix()
#     # if conda_prefix:
#     #     header_path = os.path.join(conda_prefix, "include")
#     #     lib_path = os.path.join(conda_prefix, "lib")
#     if "CMPLR_ROOT" in os.environ:
#         cmplr_root = os.environ["CMPLR_ROOT"]
#         lib_path = os.path.join(cmplr_root, "linux/lib")
#         header_path = os.path.join(cmplr_root, "linux/include")
#     elif "LD_LIBRARY_PATH" in os.environ:
#         ldlib_path = os.environ["LD_LIBRARY_PATH"]
#         paths = ldlib_path.split(":")
#         lib_path = None
#         for path in paths:
#             if "compiler" in path and "lib" in path:
#                 lib_path = path
#                 break
#         if lib_path is not None:
#             head = lib_path.split("lib")[0]
#             header_path = os.path.join(head, "include")
#         else:
#             raise RuntimeError("SYCL lib path not found in LD_LIBRARY_PATH.")
#     else:
#         raise RuntimeError(
#             "Neither of CONDA_PREFIX, LD_LIBRARY_PATH nor CMPLR_ROOT is configured."
#         )
#     if header_path and lib_path:
#         if os.path.exists(header_path) and os.path.exists(lib_path):
#             return header_path, lib_path
#         else:
#             raise FileNotFoundError(
#                 "Header ('{0:s}') and lib ('{1:s}') don't exist.".format(
#                     header_path, lib_path
#                 )
#             )
#     else:
#         raise RuntimeError(
#             "Header path ({0:s}) and library path ({1:s}) don't exist.".format(
#                 header_path, lib_path
#             )
#         )


# def find_mkl_header_path():
#     header_path, lib_path = None, None
#     # conda_prefix = get_conda_prefix()
#     # if conda_prefix:
#     #     header_path = os.path.join(conda_prefix, "include")
#     #     lib_path = os.path.join(conda_prefix, "lib")
#     if "MKLROOT" in os.environ:
#         mklroot = os.environ["MKLROOT"]
#         lib_path = os.path.join(mklroot, "lib/intel64")
#         header_path = os.path.join(mklroot, "include")
#     elif "LD_LIBRARY_PATH" in os.environ:
#         ldlib_path = os.environ["LD_LIBRARY_PATH"]
#         paths = ldlib_path.split(":")
#         lib_path = None
#         for path in paths:
#             if "mkl" in path and "lib" in path:
#                 lib_path = path
#                 break
#         if lib_path is not None:
#             head = lib_path.split("lib")[0]
#             header_path = os.path.join(head, "include")
#         else:
#             raise RuntimeError("MKL lib path not found in LD_LIBRARY_PATH.")
#     else:
#         raise RuntimeError(
#             "Neither of CONDA_PREFIX, LD_LIBRARY_PATH nor CMPLR_ROOT is configured."
#         )
#     if header_path and lib_path:
#         if os.path.exists(header_path) and os.path.exists(lib_path):
#             return header_path, lib_path
#         else:
#             raise FileNotFoundError(
#                 "Header ('{0:s}') and lib ('{1:s}') don't exist.".format(
#                     header_path, lib_path
#                 )
#             )
#     else:
#         raise RuntimeError(
#             "Header path ({0:s}) and library path ({1:s}) don't exist.".format(
#                 header_path, lib_path
#             )
#         )


# def get_ext_modules():
#     library_dirs = []
#     runtime_library_dirs = []

#     sycl_header, sycl_lib = find_sycl_header_path()
#     mkl_header, mkl_lib = find_mkl_header_path()

#     library_dirs.append(os.path.dirname(dpctl.__file__))

#     if IS_LINUX:
#         runtime_library_dirs.append(os.path.dirname(dpctl.__file__))

#     ext_dpexrt_python = Extension(
#         name="numba_dpex.core.runtime._dpexrt_python",
#         sources=[
#             "numba_dpex/core/runtime/_dpexrt_python.c",
#             "numba_dpex/core/runtime/_nrt_helper.c",
#             "numba_dpex/core/runtime/_nrt_python_helper.c",
#         ],
#         libraries=["DPCTLSyclInterface"],
#         library_dirs=library_dirs,
#         runtime_library_dirs=runtime_library_dirs,
#         include_dirs=[
#             sysconfig.get_paths()["include"],
#             numba.extending.include_path(),
#             numpy.get_include(),
#             dpctl.get_include(),
#         ],
#     )

#     library_dirs.append(mkl_lib)
#     ext_onemkl_lapack = Extension(  # noqa: F841
#         name="numba_dpex.onemkl._dpex_lapack_iface",
#         sources=["numba_dpex/onemkl/_dpex_lapack_iface.cpp"],
#         libraries=[
#             "mkl_sycl",
#             "mkl_intel_ilp64",
#             "mkl_tbb_thread",
#             "mkl_core",
#             "sycl",
#             "OpenCL",
#             "pthread",
#             "m",
#             "dl",
#         ],
#         library_dirs=library_dirs,
#         runtime_library_dirs=runtime_library_dirs,
#         include_dirs=[
#             sysconfig.get_paths()["include"],
#             numba.extending.include_path(),
#             numpy.get_include(),
#             dpctl.get_include(),
#             sycl_header,
#             os.path.join(sycl_header, "sycl"),
#             mkl_header,
#         ],
#     )

#     ext_modules = [ext_dpexrt_python]  # , ext_onemkl_lapack]

#     return ext_modules


# # icx -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2
# #     -Wall -Wformat -Wformat-security -fstack-protector-all
# #     -D_FORTIFY_SOURCE=2 -fpic -fPIC -O2
# #     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
# #     -Wformat -Wformat-security -fstack-protector-all -D_FORTIFY_SOURCE=2 -fpic -fPIC -O2
# #     -march=nocona -mtune=haswell -ftree-vectorize -fPIC
# #     -fstack-protector-strong -fno-plt -O2 -ffunction-sections
# #     -pipe -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
# #     -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
# #     -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
# #     -fPIC -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include/python3.9
# #     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages
# #     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/numpy/core/include
# #     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/dpctl/include
# #     -I/localdisk/intel/oneapi/compiler/2023.1.0/linux/include
# #     -I/localdisk/intel/oneapi/mkl/2023.1.0/include
# #     -I/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include/python3.9
# #     -c numba_dpex/onemkl/_dpex_lapack_iface.cpp
# #     -o build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o

# # icpx -shared -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
# #     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
# #     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
# #     -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro
# #     -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections
# #     -Wl,-rpath,/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
# #     -Wl,-rpath-link,/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
# #     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib
# #     -march=nocona -mtune=haswell -ftree-vectorize -fPIC
# #     -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe
# #     -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
# #     -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
# #     -isystem /localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/include
# #     build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o
# #     -L/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/dpctl
# #     -L/localdisk/intel/oneapi/mkl/2023.1.0/lib/intel64
# #     -Wl,--enable-new-dtags,-R/localdisk/work/akmkhale/opt/anaconda3/envs/ndpx/lib/python3.9/site-packages/dpctl
# #     -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread
# #     -lm -ldl
# #     -o build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so

# # -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..
# # -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..

# # -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..
# # -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../..


# def compile_onemkl_lapack_extension():
#     conda_prefix = get_conda_prefix()
#     sycl_header, sycl_lib = find_sycl_header_path()
#     mkl_header, mkl_lib = find_mkl_header_path()

#     print(conda_prefix)
#     compile_cmd = """
#     icx -Wno-unused-result -Wsign-compare -Wall -Wformat -Wformat-security
#         -fwrapv -fstack-protector-all -ftree-vectorize -fPIC
#         -fstack-protector-strong -fno-plt -ffunction-sections
#         -pipe -march=nocona -mtune=haswell
#         -isystem {0:s}/include
#         -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
#         -I{0:s}/include
#         -I{0:s}/include/python3.9
#         -I{0:s}/lib/python3.9/site-packages
#         -I{0:s}/lib/python3.9/site-packages/numpy/core/include
#         -I{0:s}/lib/python3.9/site-packages/dpctl/include
#         -I{1:s}
#         -I{2:s}
#         -c numba_dpex/onemkl/_dpex_lapack_iface.cpp
#         -o build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o
#     """.format(
#         conda_prefix, sycl_header, mkl_header
#     )

#     link_cmd = """
#     icpx -shared -pipe
#         -Wl,-O2
#         -Wl,--sort-common
#         -Wl,--as-needed
#         -Wl,-z,relro
#         -Wl,-z,now
#         -Wl,--disable-new-dtags
#         -Wl,--gc-sections
#         -Wl,-rpath,{0:s}/lib
#         -Wl,-rpath-link,{0:s}/lib
#         -Wl,--enable-new-dtags,-R{0:s}/lib/python3.9/site-packages/dpctl
#         -march=nocona -mtune=haswell
#         -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -ffunction-sections
#         -isystem {0:s}/include
#         -DNDEBUG -D_FORTIFY_SOURCE=2 -O2
#         build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.o
#         -L{0:s}/lib
#         -L{0:s}/lib/python3.9/site-packages/dpctl
#         -L{1:s}
#         -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread
#         -lm -ldl
#         -o build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so
#     """.format(
#         conda_prefix, mkl_lib
#     )

#     print("compiling numba_dpex/onemkl/_dpex_lapack_iface.cpp ...")

#     compile_cmd = re.sub(r"\n+", " ", re.sub(r"\s+", " ", compile_cmd)).strip()
#     os.makedirs(
#         "./build/temp.linux-x86_64-cpython-39/numba_dpex/onemkl", exist_ok=True
#     )

#     fout = open("/tmp/compile.sh", "w")
#     fout.write("#/bin/sh\n")
#     fout.write(compile_cmd + "\n")
#     # st = os.stat("/tmp/compile.sh")
#     # os.chmod("/tmp/compile.sh", st.st_mode | stat.S_IEXEC)
#     fout.close()

#     try:
#         output = subprocess.check_call(
#             ["sh /tmp/compile.sh"],
#             stderr=subprocess.DEVNULL,
#             stdout=subprocess.DEVNULL,
#             shell=True,
#         )
#     except subprocess.CalledProcessError as exc:
#         print("Status : FAIL", exc.returncode, exc.output)
#     else:
#         print("Output: \n{}\n".format(output))

#     print(
#         "linking build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so ..."
#     )

#     link_cmd = re.sub(r"\n+", " ", re.sub(r"\s+", " ", link_cmd)).strip()
#     os.makedirs(
#         "./build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl", exist_ok=True
#     )

#     fout = open("/tmp/link.sh", "w")
#     fout.write("#/bin/sh\n")
#     fout.write(link_cmd + "\n")
#     # st = os.stat("/tmp/link.sh")
#     # os.chmod("/tmp/link.sh", st.st_mode | stat.S_IEXEC)
#     fout.close()

#     subprocess.check_call(
#         ["sh /tmp/link.sh"],
#         stderr=subprocess.DEVNULL,
#         stdout=subprocess.DEVNULL,
#         shell=True,
#     )

#     print(
#         "copying ./build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so -> ./numba_dpex/onemkl"
#     )
#     shutil.copy(
#         "./build/lib.linux-x86_64-cpython-39/numba_dpex/onemkl/_dpex_lapack_iface.cpython-39-x86_64-linux-gnu.so",
#         "./numba_dpex/onemkl",
#     )


# def _llvm_spirv():
#     """Return path to llvm-spirv executable."""

#     try:
#         import dpcpp_llvm_spirv as dls
#     except ImportError:
#         raise ImportError("Cannot import dpcpp-llvm-spirv package")

#     result = dls.get_llvm_spirv_path()

#     return result


# def spirv_compile():
#     clang_args = [
#         get_spriv_compiler(),
#         "-flto",
#         "-fveclib=none",
#         "-target",
#         "spir64-unknown-unknown",
#         "-c",
#         "-x",
#         "cl",
#         "-emit-llvm",
#         "-cl-std=CL2.0",
#         "-Xclang",
#         "-finclude-default-header",
#         "numba_dpex/ocl/atomics/atomic_ops.cl",
#         "-o",
#         "numba_dpex/ocl/atomics/atomic_ops.bc",
#     ]

#     spirv_args = [
#         _llvm_spirv(),
#         "--spirv-max-version",
#         "1.4",
#         "numba_dpex/ocl/atomics/atomic_ops.bc",
#         "-o",
#         "numba_dpex/ocl/atomics/atomic_ops.spir",
#     ]

#     subprocess.check_call(
#         clang_args,
#         stderr=subprocess.DEVNULL,
#         stdout=subprocess.DEVNULL,
#         shell=False,
#     )

#     subprocess.check_call(
#         spirv_args,
#         stderr=subprocess.DEVNULL,
#         stdout=subprocess.DEVNULL,
#         shell=False,
#     )


# class install(orig_install.install):
#     def run(self):
#         spirv_compile()
#         # compile_onemkl_lapack_extension()
#         super().run()


# class develop(orig_develop.develop):
#     def run(self):
#         spirv_compile()
#         # compile_onemkl_lapack_extension()
#         super().run()


# def _get_cmdclass():
#     cmdclass = versioneer.get_cmdclass()
#     cmdclass["install"] = install
#     cmdclass["develop"] = develop
#     return cmdclass


# packages = find_packages(
#     include=[
#         "numba_dpex",
#         "numba_dpex.*",
#         "_dpexrt_python",
#         "_dpex_lapack_iface",
#     ]
# )
# install_requires = [
#     "numba >={}".format("0.57"),
#     "dpctl",
#     "packaging",
# ]

# metadata = dict(
#     name="numba-dpex",
#     version=versioneer.get_version(),
#     cmdclass=_get_cmdclass(),
#     description="An extension for Numba to add data-parallel offload capability",
#     url="https://github.com/IntelPython/numba-dpex",
#     packages=packages,
#     install_requires=install_requires,
#     include_package_data=True,
#     zip_safe=False,
#     ext_modules=get_ext_modules(),
#     author="Intel Corporation",
#     classifiers=[
#         "Development Status :: 4 - Beta",
#         "Environment :: GPU",
#         "Environment :: Plugins",
#         "Intended Audience :: Developers",
#         "License :: OSI Approved :: Apache 2.0",
#         "Operating System :: OS Independent",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: Implementation :: CPython",
#         "Topic :: Software Development :: Compilers",
#     ],
#     entry_points={},
# )

# # cc_orig, cxx_orig = None, None
# # if "CC" in os.environ:
# #     cc_orig = os.environ["CC"]
# # if "CXX" in os.environ:
# #     cxx_orig = os.environ["CXX"]

# # os.environ["CC"] = "icx"
# # os.environ["CXX"] = "icpx"

# ensure_dpnp()
# setup(**metadata)

# # if cc_orig:
# #     os.environ["CC"] = cc_orig
# # if cxx_orig:
# #     os.environ["CXX"] = cxx_orig
