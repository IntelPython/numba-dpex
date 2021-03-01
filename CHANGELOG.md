# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0] - 2021-03-01
### Added
- Documentation.
- Add support for dpctl.dparray.
- Support NumPy functions via DPNP: random, linalg, transcendental, array ops, array creation.
- Wheels building.
- Using Bandit for finding common security issues in Python code.

### Changed
- Start using black code style formatter.
- Build SPIRV code in setup.py.
- Start using pytest for running tests.
- Start using Apache 2.0 license.
- Consistency of file headers.
- Updated to Numba 0.52, dpCtl 0.6 and dpNP 0.5.1.
- Don't create a new copy of a usm shared array data pointers for kernel call.
- Modify test cases and examples to use Level Zero queue.

### Fixed
- Fix incorrect import in examples.

## [0.12.0] - 2020-12-17
### Added
- numba-dppy is a standalone package now. Added setup.py and conda recipe.
- Offload diagnostics.
- Controllable fallback.
- Add flags to generate debug symbols.
- Implementation of `np.linalg.eig`, `np.ndarray.sum`, `np.ndarray.max`, `np.ndarray.min`, `np.ndarray.mean`.
- Two new re-write passes to convert NumPy calls into a pseudo `numba_dppy` call site to allow target-specific
  overload of NumPy functions. The rewrite passes is a temporary fix till Numba gains support for target-specific overlaods.
- Updated to dpCtl 0.5.* and dpNP 0.4.*

### Changed
- The `dpnp` interface now uses Numba's `@overload` functionality as opposed to the previous `@lower_builtin` method.
- Rename `DPPL` to `DPPY`.
- Cleaned test code.
- `DPPLTestCase` replaced with `unittest.TestCase`.
- All tests and examples use `with device_context`.
- Config environment variables starts with `NUMBA_DPPY_`
(i.e. NUMBA_DPPY_SAVE_IR_FILES and NUMBA_DPPY_SPIRV_VAL)
- Remove nested folder `dppl` in `tests`.
- No dependency on `cffi`.

### Removed
- The old backup file.

## NUMBA Version 0.48.0 + DPPY Version 0.3.0 (June 29, 2020)

This release includes:
- Caching of dppy.kernels which will improve performance.
- Addition of support for Intel Advisor which will help in profiling applications.
