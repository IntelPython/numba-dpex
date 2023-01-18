[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/IntelPython/numba-dpex/badge.svg?branch=main)](https://coveralls.io/github/IntelPython/numba-dpex?branch=main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo" width="75"/>
<br/>
<br/>
<br/>
<br/>


# What?

The data-parallel extension to Numba (Numba-dpex) adds data-parallel kernel
programming and auto-offload capabilities to [Numba](http://numba.pydata.org).
The extension uses the Intel&reg; DPC++ runtime and supports various OpenCL and
Level Zero devices. Refer the [user
guide](https://intelpython.github.io/numba-dpex/) for more details.

# Installing

Numba-dpex is part of the Intel&reg; Distribution of Python (IDP) and Intel&reg;
oneAPI BaseKit, and can be installed along with oneAPI. Additionally, we support
installing it from Anaconda cloud and PyPi. Please refer the instructions
on our [documentation page](https://intelpython.github.io/numba-dpex/latest/user_guides/getting_started.html)
for more details.

# Getting started

A good starting point is to run the test suite that includes the unit tests
inside the **numba_dpex/tests** module. To run the tests, invoke:

```bash
python -m pytest --pyargs numba_dpex.tests
```
or
```bash
pytest
```
Once you run the tests and make sure your numba-dpex installation is up and
running, try out the examples inside the **numba_dpex/examples** folder. For
example, you can try the `vector addition` example as follows:
```bash
python numba_dpex/examples/vector_sum.py
```

# Known Issue
Floor division operator `//` is not supported inside the `numba_dpex.kernel`
kernel programming API. The below code snippet will result in error reported in
this [Issue](https://github.com/IntelPython/numba-dpex/issues/571).

```
import numpy as np, numba_dpex
@numba_dpex.kernel
def div_kernel(dst, src, m):
    i = numba_dpex.get_global_id(0)
    dst[i] = src[i] // m

import dpctl
with dpctl.device_context(dpctl.SyclQueue()):
    X = np.arange(10)
    Y = np.arange(10)
    div_kernel[10, numba_dpex.DEFAULT_LOCAL_SIZE](Y, X, 5)
    D = X//5
    print(Y, D)
```

To bypass this issue, the `llvm-spirv` tool from the `dpcpp` conda package needs
to be used.

For linux: `conda install dpcpp_linux-64`
For Windows: `conda install dpcpp_win-64`

# Learn more?

Detailed documentation including user guides are hosted on our
[documentation site](https://intelpython.github.io/numba-dpex).

# Found a bug?

Please report issues and bugs directly on
[github](https://github.com/IntelPython/numba-dpex/issues).

## Test Matrix:

|   #   |   OS    | Distribution |  Python  |  Architecture   | Test type |  IntelOneAPI   | Build Commands |    Dependencies    |   Backend   |
| :---: | :-----: | :----------: | :------: | :-------------: | :-------: | :------------: | :------------: | :----------------: | :---------: |
|   1   |  Linux  | Ubuntu 20.04 | 3.7, 3.8 | Gen9 Integrated |    CI     | 2021.3, 2021.4 |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.1 |
|   2   |  Linux  | Ubuntu 20.04 | 3.7, 3.8 | Gen12 Discrete  |  Manual   | 2021.3, 2021.4 |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.1 |
|   3   |  Linux  | Ubuntu 20.04 | 3.7, 3.8 |    i7-10710U    |    CI     | 2021.3, 2021.4 |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.1 |
|   4   | Windows |      10      | 3.7, 3.8 | Gen9 Integrated |    CI     | 2021.3, 2021.4 |      (1)       |    Numba, NumPy    |     OCL     |
|   5   | Windows |      10      | 3.7, 3.8 |    i7-10710     |    CI     | 2021.3, 2021.4 |      (1)       |    Numba, NumPy    |     OCL     |

(1): `python setup.py install; pytest -q -ra --disable-warnings --pyargs numba_dpex -vv`
