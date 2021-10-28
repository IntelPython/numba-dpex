[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo" width="75"/>
<br/>
<br/>
<br/>
<br/>


# What?

Numba-dppy is an extension to the [Numba](http://numba.pydata.org) Python JIT
compiler that provides a way to write data-parallel kernels directly in Python
and offload them on various types of Intel&reg; architectures including CPUs,
integrated GPUs and discrete GPUs. The compiler also supports offloading NumPy
function calls and Numba `prange` loops. Refer the
[user guide](https://intelpython.github.io/numba-dppy/) for more details.

# Installing

Numba-dppy is part of the Intel&reg; Distribution of Python (IDP) and Intel&reg;
oneAPI BaseKit, and can be installed along with oneAPI. Additionally, we support
installing it from Anaconda cloud and PyPi. Please refer the instructions
on our [documentation page](https://intelpython.github.io/numba-dppy/latest/user_guides/getting_started.html)
for more details.

# Give it a try?

A good starting point is to run the test suite that includes the unit tests
inside the `numba_dppy/tests` module. To run the tests, invoke:

```bash
python -m pytest --pyargs numba_dppy.tests
```
or
```bash
pytest
```
Once you run the tests and make sure your Numba-dppy installation is up and
running, try out the examples inside the `numba_dppy/examples` folder. For
example, you can try the `vector addition` example as follows:
```bash
python numba_dppy/examples/sum.py
```

# Learn more?

Detailed documentation including user guides are hosted on our
[documentation site](https://intelpython.github.io/numba-dppy).

# Found a bug?

Please report issues and bugs directly on
[github](https://github.com/IntelPython/numba-dppy/issues).

## Test Matrix:

|   #   |   OS    | Distribution |  Python  |  Architecture   | Test type |  IntelOneAPI   | Build Commands |    Dependencies    |   Backend   |
| :---: | :-----: | :----------: | :------: | :-------------: | :-------: | :------------: | :------------: | :----------------: | :---------: |
|   1   |  Linux  | Ubuntu 20.04 | 3.7, 3.8 | Gen9 Integrated |    CI     | 2021.3, 2021.4 |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.1 |
|   2   |  Linux  | Ubuntu 20.04 | 3.7, 3.8 | Gen12 Discrete  |  Manual   | 2021.3, 2021.4 |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.1 |
|   3   |  Linux  | Ubuntu 20.04 | 3.7, 3.8 |    i7-10710U    |    CI     | 2021.3, 2021.4 |      (1)       | Numba, NumPy, dpnp | OCL, L0-1.1 |
|   4   | Windows |      10      | 3.7, 3.8 | Gen9 Integrated |    CI     | 2021.3, 2021.4 |      (1)       |    Numba, NumPy    |     OCL     |
|   5   | Windows |      10      | 3.7, 3.8 |    i7-10710     |    CI     | 2021.3, 2021.4 |      (1)       |    Numba, NumPy    |     OCL     |

(1): `python setup.py install; pytest -q -ra --disable-warnings --pyargs numba_dppy -vv`
