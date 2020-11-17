import os
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize


def get_ext_modules():
    ext_modules = []

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
        ext_dpnp_glue = Extension(name='numba_dppy.dpnp_glue.dpnp_fptr_interface',
                                  sources=['numba_dppy/dpnp_glue/dpnp_fptr_interface.pyx'],
                                  include_dirs=[dpnp.get_include()],
                                  libraries=['dpnp_backend_c'],
                                  library_dirs=dpnp_lib_path,
                                  runtime_library_dirs=dpnp_lib_path,
                                  language="c++")
        ext_modules += [ext_dpnp_glue]

    if dpnp_present:
        return cythonize(ext_modules)
    else:
        return ext_modules


packages = find_packages(include=["numba_dppy", "numba_dppy.*"])
build_requires = ["cython"]
install_requires = [
    "numba",
    "dpctl",
]

metadata = dict(
    name="numba-dppy",
    version="0.0.1",
    description="Numba extension for Intel CPU and GPU backend",
    url="https://github.com/IntelPython/numba-dppy",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
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
)

setup(**metadata)
