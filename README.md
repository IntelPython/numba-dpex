[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# numba-dppy

## Numba + dpPy + dpCtl + dpNP = numba-dppy

`numba-dppy` extends Numba with a new backend to support compilation
for Intel CPU and GPU architectures.

For more information about Numba, see the Numba homepage:
http://numba.pydata.org.

Note: `numba-dppy` requires patched version of Numba.
See https://github.com/IntelPython/numba.

For more information about dpCtl, see the dpCtl homepage:
https://intelpython.github.io/dpctl/

For more information about dpNP, see the dpNP homepage:
https://intelpython.github.io/dpnp/

## Dependencies

* numba >=0.51 (IntelPython/numba)
* dpCtl >=0.5.1
* dpNP 0.4.* (optional)
* llvm-spirv (SPIRV generation from LLVM IR)
* llvmdev (LLVM IR generation)
* spirv-tools
* scipy (for testing)

## dpPy

dpPy is a proof-of-concept backend for NUMBA to support compilation for
Intel CPU and GPU architectures.
The present implementation of dpPy is based on OpenCL 2.1, but is likely
to change in the future to rely on Sycl/DPC++ or Intel Level-0 driver API.

## Installation

Use setup.py or conda (see conda-recipe).

## Testing

See folder `numba_dppy/tests`.

Run tests:
```bash
python -m unittest numba_dppy.tests
```

## Examples

See folder `numba_dppy/examples`.

Run examples:
```bash
python numba_dppy/examples/sum.py
```

## How Tos

Refer the [HowTo.rst](docs/HowTo.rst) guide for an overview of the programming semantics,
examples, supported functionalities, and known issues.

## Debugging

Please follow instructions in the [DEBUGGING.md](docs/DEBUGGING.md)

## Reporting issues

Please use https://github.com/IntelPython/numba-dppy/issues to report issues and bugs.

## Features

Read this guide for additional features [INDEX.md](docs/INDEX.md)
