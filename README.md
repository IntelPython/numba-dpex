[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/IntelPython/numba-dpex/badge.svg?branch=main)](https://coveralls.io/github/IntelPython/numba-dpex?branch=main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Join the chat at https://matrix.to/#/#Data-Parallel-Python_community:gitter.im](https://badges.gitter.im/Join%20Chat.svg)](https://app.gitter.im/#/room/#Data-Parallel-Python_community:gitter.im)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/29068/badge.svg)](https://scan.coverity.com/projects/intelpython-numba-dpex)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/{owner}/{repo}/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/numba-dpex
<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo" width="75"/>
<br/>
<br/>
<br/>
<br/>



Data-parallel Extension for Numba* (numba-dpex) is a standalone extension for
the [Numba](http://numba.pydata.org) Python JIT compiler. Numba-dpex provides
a generic kernel programming API and an offload feature that extends Numba's
auto-parallelizer to generate data-parallel kernels for `parfor` nodes.

Numba-dpex's kernel API has a design and API similar to Numba's `cuda.jit`
module, but is based on the [SYCL](https://sycl.tech/) language. The
code-generation for the kernel API currently supports
[SPIR-V](https://www.khronos.org/spir/)-based
[OpenCL](https://www.khronos.org/opencl/) and
[oneAPI Level Zero](https://spec.oneapi.io/level-zero/latest/index.html)
devices that are supported by Intel&reg; DPC++ SYCL compiler runtime. Supported
devices include Intel&reg; CPUs, integrated GPUs and discrete GPUs.

The offload functionality in numba-dpex is based on Numba's `parfor`
loop-parallelizer. Our compiler extends Numba's `parfor` feature to generate
kernels and offload them to devices supported by DPC++ SYCL compiler runtime.
The offload functionality is supported via a new NumPy drop-in replacement
library: [dpnp](https://github.com/IntelPython/dpnp). Note that `dpnp` and NumPy-based
expressions can be used together in the same function, with `dpnp` expressions getting
offloaded by `numba-dpex` and NumPy expressions getting parallelized by Numba.

Refer the [documentation](https://intelpython.github.io/numba-dpex) and examples
to learn more.

# Getting Started

Numba-dpex is part of the Intel&reg; Distribution of Python (IDP) and Intel&reg;
oneAPI AIKit, and can be installed along with oneAPI. Additionally, we support
installing it from Anaconda cloud. Please refer the instructions
on our [documentation page](https://intelpython.github.io/numba-dpex/latest/getting_started.html)
for more details.

Once the package is installed, a good starting point is to run the examples in
the `numba_dpex/examples` directory. The test suite may also be invoked as
follows:

```bash
python -m pytest --pyargs numba_dpex.tests
```

## Conda

To install `numba_dpex` from the Intel(R) channel on Anaconda
cloud, use the following command:

```bash
conda install numba-dpex -c intel -c conda-forge
```

## Pip

The `numba_dpex` can be installed using `pip` obtaining wheel packages either from PyPi or from Intel(R) channel on Anaconda.
To install `numba_dpex` wheel package from Intel(R) channel on Anaconda, run the following command:

```bash
python -m pip install --index-url https://pypi.anaconda.org/intel/simple numba-dpex
```

# Contributing

Please create an issue for feature requests and bug reports. You can also use
the GitHub Discussions feature for general questions.

If you want to chat with the developers, join the
[#Data-Parallel-Python_community](https://app.gitter.im/#/room/#Data-Parallel-Python_community:gitter.im) room on Gitter.im.

Also refer our [CONTRIBUTING](https://github.com/IntelPython/numba-dpex/blob/main/CONTRIBUTING.md) page.
