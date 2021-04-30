[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# numba-dppy

## Numba + dppy + dpctl + dpnp = numba-dppy

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

* numba 0.53.* (IntelPython/numba)
* dpctl 0.7.*
* dpnp >=0.5.1 (optional)
* llvm-spirv (SPIRV generation from LLVM IR)
* llvmdev (LLVM IR generation)
* spirv-tools
* scipy (for testing)

## dppy

dppy is a proof-of-concept backend for Numba to support compilation for
Intel CPU and GPU architectures.
The present implementation of dpPy is based on OpenCL 2.1, but is likely
to change in the future to rely on Sycl/DPC++ or Intel Level-0 driver API.

## Installation

Use setup.py or conda (see conda-recipe).

## Testing

See folder `numba_dppy/tests`.

Run tests:
```bash
python -m pytest --pyargs numba_dppy.tests
```
or
```bash
pytest
```

## Examples

See folder `numba_dppy/examples`.

Run examples:
```bash
python numba_dppy/examples/sum.py
```

## Debugging

Please follow instructions in the [debugging.md](docs/user_guides/debugging.md)

## Reporting issues

Please use https://github.com/IntelPython/numba-dppy/issues to report issues and bugs.

## Features

Read this guide for additional features [INDEX.md](docs/INDEX.md)

## Test Matrix:

|   #   |   OS    | Distribution | Python |  Architecture   | Test type | IntelOneAPI | Build Commands |    Dependencies    |   Backend   |
| :---: | :-----: | :----------: | :----: | :-------------: | :-------: | :---------: | :------------: | :----------------: | :---------: |
|   1   |  Linux  | Ubuntu 20.04 |  3.7   | Gen9 Integrated |    CI     |   2021.2    |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.0 |
|   2   |  Linux  | Ubuntu 20.04 |  3.7   | Gen12 Discrete  |  Manual   |   2021.2    |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.0 |
|   3   |  Linux  | Ubuntu 20.04 |  3.7   |    i7-10710U    |    CI     |   2021.2    |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.0 |
|   4   | Windows |      10      |  3.7   | Gen9 Integrated |    CI     |   2021.2    |      (1)       |    Numba, NumPy    |     OCL     |
|   5   | Windows |      10      |  3.7   |    i7-10710     |    CI     |   2021.2    |      (1)       |    Numba, NumPy    |     OCL     |

(1): `python setup.py install; pytest -q -ra --disable-warnings --pyargs numba_dppy -vv`
