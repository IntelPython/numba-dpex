from setuptools import find_packages, setup

packages = find_packages(include=["numba_dppy", "numba_dppy.*"])

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
    install_requires=install_requires,
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
