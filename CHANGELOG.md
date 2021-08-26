# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.15.0] - 2021-08-25

### Added
- Introduce array ultilites to check, allocate and copy memory using SYCL USM (#489)
- Add packaging in run dependencies (#505)
- Add skipping tests for run without GPU (#508)
- Add CI pipeline on GitHub Actions for Linux (#507)
- Enable dpctl.tensor.usm_ndarray for @dppy.kernel (#509)
- Enable @vectorize for target dppy (#497)
- Add integration channels to GitHub Actions and make workflow consistent with dpctl (#510)

### Changed
- Update to Numba 0.54.0 (#493)
- Update to dpctl 0.9 (#514)
- Update to dpnp 0.7 (#513)
- Use dpcpp compiler package for Linux (#502)
- Update go version (#515)

### Removed
- Remove llvmdev from runtime dependecies (#498)

### Fixed
- Fix required compiler flags for processing genreated DWARF symbols (#500)

## [0.14.4] - 2021-07-09

### Fixed
- Fix emission of debug information (#424)
- Fix examples demonstrating GDB usage on a numba_dppy.kernel function. (#455)
- Remove address space cast from global to generic inside numba_dppy.kernel (#432)
- Fix debugging of local variables (#432)
- Assert offload to SYCL devices in tests (#466)
- Removed calling opt to convert LLVM IR to LLVM Bitcode (#481)

### Added
- Add examples for debugging (#426)
- Added a new NUMBA_DPPY_DEBUGINFO environment variable to control GDB usage (#460)
- Add debug option to dppy.kernel decorator (#424)
- Enable debugging of nested GPU functions (#424)
- Enable setting breakpoints by function names while Debugging (#434)
- Various fixes and improvements to documentation about debugging
  (#479, #474, #475, #480, #475, #477, #468,#450)
- Add automatic generation of commands for debugging (#463)
- Add tests on debugging local variables (#421)
- Enable eager compilation of numba_dppy.kernel (#291)
- Increase test coverage for native atomic ops (#435)
- Check and deter users from returning values from numba_dppy.kernel (#476)

## [0.14.3] - 2021-05-27

### Fixed

- Add check for ONEAPI_ROOT dir (#411)
- Fix using unquoted environment variable for clang path (#386)
- Fix kernel caching (#408)

## [0.14.2] - 2021-05-26

### Added
- Update documentation: version 0.14 (#378), API docs (#388),
  note about Intel Python Numba (#389),
- Update User Guides about Debugging (#380), recommendations (#323),
  locals (#394), stepping (#400), configure environment (#402),
  set up machine (#396), info functions (#405)
- Update Developer Guides about DPNP integration (#362)
- Update README: add link to docs (#379), add Cython and pytest in dependencies,
  add test matrix (#305)
- Add initial integration testing with dpnp and usm_ndarray (#403)
- Introduce type in Numba-dppy to represent dpctl.tensor.usm_ndarray (#391)
- Improve error reporting when searching for dpctl. (#368)
- Enable Python 3.8 in CI (#359)
- Adds a new utils submodule to provide LLVM IR builder helpers. (#355)
- Improve warning and error messages when parfor offload fails. (#353)
- Extend itanium mangler to support numba.types.CPointer and add test (#339)
- Enable optimization level setting (#62)
- Improve message printed during parfor lowering. (#337)
- Initial tests for debug info (#297)
- Add Bandit checks (#264)

### Changed
- Update to dpctl 0.8 (#375)
- Update to Numba 0.53 (#279), inluding
  override get_ufunc_info in DPPYTargetContext (#367)
- Update to DPNP 0.6 (#359)
- Refactor test function generation (#374)
- Ignore the cython generated cpp files inside dpnp_glue. (#351)
- Add automerge main to gold/2021 (#349)
- Fix dpnp version restriction in conda recipe (#347)
- Change all Numba-dppy examples to work wih dpctl 0.7.0 (#309)
- Restrict dpnp version (#344)
- Feature changes related to dpctl 0.7 (#340)
- Rename dpNP to dpnp (#334)
- Ignore generated spir files (#333)
- Use correct names for other products dpctl, Numba, dpnp (#310)
- Remove dead code only if function name is replaced (#303)
- Update license in conda recipe (#350)
- Update blackscholes examples (#377)

### Fixed
- Fix dppy_rt extension (#393)
- Update SYCL Filter String (#390)
- Fix atomics (#346)
- Fixes memory leaks in the usage of dpctl C API functions. (#369)
- Fix SPIR-V validation (#354)
- Fix run pre-commit check on main branch
- Fix tests to skip if device is not available (#345)
- Make Test Matrix table smaller in README (#308)
- Fix black action. (#306)
- Fix "subprocess.check_call" for Windows (#269)

## [0.13.1] - 2021-03-11
### Fixed
- Add spir file to package.
- Do not modify CC env variable during build.
- Use correct version in documentation.

## [0.13.0] - 2021-03-02
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
