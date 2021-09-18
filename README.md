[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# numba-dppy

## Numba + dppy + dpctl + dpnp = numba-dppy

`numba-dppy` extends Numba with a new backend to support compilation
for Intel CPU and GPU architectures.

For more information about Numba, see the Numba homepage:
http://numba.pydata.org.

For more information about dpCtl, see the dpCtl homepage:
https://intelpython.github.io/dpctl/

For more information about dpNP, see the dpNP homepage:
https://intelpython.github.io/dpnp/

## Dependencies

* numba 0.54.*
* dpctl 0.10.*
* dpnp 0.8.* (optional)
* llvm-spirv 11.* (SPIRV generation from LLVM IR)
* spirv-tools
* packaging
* cython (for building)
* pytest (for testing)
* scipy (for testing)

## dppy

dppy is a proof-of-concept backend for Numba to support compilation for
Intel CPU and GPU architectures.
The present implementation of dpPy is based on OpenCL 2.1, but is likely
to change in the future to rely on Sycl/DPC++ or Intel Level-0 driver API.

## Installation

Please follow the instructions provided [here](https://intelpython.github.io/numba-dppy/latest/user_guides/getting_started.html).

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

## Documentation

Detailed documentation including user guides is hosted at [https://intelpython.github.io/numba-dppy](https://intelpython.github.io/numba-dppy).

## Debugging

Please follow instructions in the [debugging manual](https://intelpython.github.io/numba-dppy/latest/user_guides/debugging/).

## Reporting issues

Please use [this link](https://github.com/IntelPython/numba-dppy/issues) to report issues and bugs.

## Features

For a detailed description of features currently supported by numba-dppy, refer the documentation hosted at
[https://intelpython.github.io/numba-dppy](https://intelpython.github.io/numba-dppy).

## Test Matrix:

|   #   |   OS    | Distribution | Python |  Architecture   | Test type | IntelOneAPI | Build Commands |    Dependencies    |   Backend   |
| :---: | :-----: | :----------: | :----: | :-------------: | :-------: | :---------: | :------------: | :----------------: | :---------: |
|   1   |  Linux  | Ubuntu 20.04 |  3.7   | Gen9 Integrated |    CI     |   2021.2    |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.0 |
|   2   |  Linux  | Ubuntu 20.04 |  3.7   | Gen12 Discrete  |  Manual   |   2021.2    |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.0 |
|   3   |  Linux  | Ubuntu 20.04 |  3.7   |    i7-10710U    |    CI     |   2021.2    |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.0 |
|   4   | Windows |      10      |  3.7   | Gen9 Integrated |    CI     |   2021.2    |      (1)       |    Numba, NumPy    |     OCL     |
|   5   | Windows |      10      |  3.7   |    i7-10710     |    CI     |   2021.2    |      (1)       |    Numba, NumPy    |     OCL     |

(1): `python setup.py install; pytest -q -ra --disable-warnings --pyargs numba_dppy -vv`
