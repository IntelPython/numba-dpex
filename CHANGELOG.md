# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.23.0] - 2024-04-XX

### Fixed
* Array alignment problem for stack arrays allocated for kernel arguments. (#1357)
* Issue #892, #906 caused by incorrect code generation for indexing (#1377)
* Fix `KernelHasReturnValueError` inside `KernelDispatcher`. (#1394)
* Issue #1390: broken support for slicing into `dpctl.tensor.usm_ndarray` in kernels (#1425)

### Added
* A new overloaded `dimensions` attribute for all index-space id classes (#1359)
* Support for `AtomicRef` creation using multi-dimensional arrays (#1367)
* Support for linearized indexing functions inside a JIT compiled kernel (#1368)
* Improved documentation: overview (#1341), kernel programming guide (#1388), API docs (#1414), configs options (#1415), comparison with SYCL API (#1417)
* New `PrivateArray` class in `kernel_api` to replace `dpex.private.array` (#1370, #1377)
* Support for libsycinterface::DPCTLKernelArgType enum for specifying type of kernel args instead of hard coding (#1382)
* New indexing unit tests for kernel_api simulator and JIT compiled modes (#1378)
* New unit tests to verify all `kernel_api` features usable inside `device_func` (#1391)
* A `sycl::local_accessor`-like API (`kernel_api.LocalAccessor`) for numba-dpex kernel (#1331)
* Specialization support for `device_func` decorator (#1398)
* Support for all `kernel_api` functions inside the `numba_dpex.kernel` decorator. (#1400)

### Changed
* Default inline threshold value set to `2` from `None`. (#1385)
* Port parfor kernel templates to `kernel_api` (#1416), (#1424)
* Minimum required dpctl version is now 0.16.1
* Minimum required numba version is now 0.59.0

### Removed
* OpenCL-like kernel API functions (#1420)
* `func` decorator (replaced by `device_func`) (#1400)
* `numba_dpex.experimental.kernel` and `numba_dpex.experimental.device_func` (#1400)

## [0.22.0] - 2024-02-19

### Fixed
* Bug in boxing a DpnpNdArray from parent (#1155)
* Strided layouts and F-contiguous layouts supported in experimental kernel (#1178)
* Barrier call code-generation on OpenCL CPU devices (#1280, #1310)
* Importing numba-dpex can break numba execution (#1267)
* Overhead on launching numba_dpex.kernel functions (#1236)

### Added
* Support for dpctl.SyclEvent data type inside dpjit (#1134)
* Support for kernel_api.Range and kernel_api.NdRange inside dpjit (#1148)
* DPEX_OPT: a numba-dpex-specific optimization level config option (#1158)
* Uploading wheels packages to anaconda (#1160)
* flake8 eradicate linter option (#1177)
* Support dpctl.SyclEvent.wait call inside dpjit (#1179)
* Creation of sycl event and queue inside dpjit (#1193, #1190, #1218)
* Experimental kernel dispatcher for kernel compilation (#1178, #1205)
* Added experimental target context for SPIRV codegen (#1213, #1225)
* GDB test cases in public CI (#1209)
* Async kernel submission option (#1219, #1249)
* A new literal type to store IntEnum as Literal types (#1227)
* SYCL-like memory enum classes to the experimental module (#1239)
* call_kernel function to launch kernels (#1260)
* Experimental overloads for an AtomicRef class and fetch_* methods (#1257, #1261)
* New device-specific USMNdArrayModel for USMNdArray and DpnpNdArray types (#1293)
* Experimental atomic load, store and exchange operations (#1297)
* Kernel_api module to simulate kernel functions in pure Python (#1304, #1326)
* Experimental implementation of group barrier operation (#1280)
* Experimental atomic compare_exchange implementation (#1312)
* Experimental group index class (#1310)
* OpenSSF scorecard (#1320)
* Experimental feature index overload methods (#1323)
* Experimental feature group index overload methods (#1330)
* API Documentation for kernel API (#1332)

### Changed
* Switch to dpc++ compiler for building numba-dpex (#1210)
* Versioneer and pytest configs into pyproject.toml (#1212)
* numba-dpex can be imported even if no SYCL device is detected by dpctl (#1272)

### Removed
* Kernel launch params as lists/tuple. Only Range/NdRange supported (#1251)
* DEFAULT_LOCAL_SIZE global constant (#1291)
* Functions to invoke spirv-tools utilities from spirv_generator (#1292)
* Incomplete vectorize decorator from numba-dpex (#1298)
* Support for Numba 0.57 (#1307)

### Deprecated
* OpenCL-like kernel API functions in numba_dpex.ocldecl module

## [0.21.4] - 2023-10-12

### Fixed
* Remove dead code to silence Coverity errors. (#1163)

## [0.21.3] - 2023-09-28

### Fixed
* Pin CI conda channels (#1133)
* Mangled kernel name generation (#1112)

### Added
* Support tests on single point precision GPUs (#1143)
* Initial work on Coverity scan CI (#1128)
* Python 3.11 support (#1123)
* Security policy (#1117)
* scikit-build to build native extensions (#1107, #1116, #1127, #1139, #1140)

### Changed
* Rename helper function to clearly indicate its usage (#1145)
* The data model used by the DpnpNdArray type for kernel functions(#1118)

### Removed
* Support for Python 3.8 (#1113)

## [0.21.2] - 2023-08-07

### Fixed
* Bugs (#1068, #774) in atomic addition caused due to improper floating point atomic emulation. (#1103)

### Changed
* Updated documentation and user guides (#1097, #879)

### Removed
* Dependency on `spirv-tools` (#1103, #1108)
* floating point atomic add emulation using `atomic_ops.cl` (#1103)
* `NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE` configuration option (#1103)

## [0.21.1] - 2023-07-17

### Changed
* Improved support for `queue` keyword in dpnp array constructor overloads (#1083)
* Improved reduction kernel example (#1089)

### Fixed
* Update Itanium CXX ABI Mangler reference (#1080)
* Update sourceware references in docstrings (#1081)
* Typo in error messages of kernel interface (#1082)

## [0.21.0] - 2023-06-17

### Added
* Support addition and multiplication-based prange reduction loops (#999)
* Proper boxing, unboxing of dpctl.SyclQueue objects inside dpjit decorated functions (#963, #1064)
* Support for `queue` keyword arguments inside dpnp array constructors in dpjit (#1032)
* Overloads for dpnp array constructors: dpnp.full (#991), dpnp.full_like (#997)
* Support for complex64 and complex128 types as kernel arguments and in parfors (#1033, #1035)
* New config to run the ConstantSizeStaticLocalMemoryPass optionally (#999)
* Support for Numba 0.57 (#1030, #1003, #1002)
* Support for Python 3.11 (#1054)
* Support for SPIRV 1.4 (#1056, #1060)

### Changed
* Parfor lowering happens using the kernel pipeline (#996)
* Minimum required Numba version is now 0.57 (#1030)
* Numba monkey patches are moved to numba_dpex.numba_patches (#1030)
* Redesigned unit test suite (#1018, #1017, #1015, #1036, #1037, #1072)

### Fixed
* Fix stride computation when unboxing a dpnp array (#1023)
* Using cached queue instead of creating new one on type inference (#946)
* Fixed bug in reduction mul operation for dpjit (#1048)
* Offload of parfor nodes to OpenCL UHD GPU devices (#1074)

### Removed
* Support for offloading NumPy-based parfor nodes to SYCL devices (#1041)
* Removed rename_numpy_functions_pass (#1041)
* Dpnp overloads using stubs (#1041, #1025)
* Support for `like` keyword argument in dpnp array constructor overloads (#1043)
* Support for NumPy arrays as kernel arguments (#1049)
* Kernel argument access specifiers (#1049)
* Support for dpctl.device_context to launch kernels and njit offloading (#1041)

## [0.20.1] - 2023-04-07

### Added
* Replaced llvm_spirv from oneAPI path by dpcpp-llvm-spirv package.(#979)
* Added Dockerfile and a manual workflow to publish pre-built packages to the repo.(#973)

### Fixed
* Fixed default dtype derivation when creating a dpnp.ndarray. (#993)
* Adjusted test_windows step to work with intel-opencl-rt=2023.1.0. (#990)
* Fixed layout in dpnp overload.(#987)
* Handled the case when arraystruct->meminfo is null to close gh-965. (#972)

## [0.20.0] - 2023-03-06

### Added
* New dpjit decorator supporting dpnp compilation (#887)
* Boxing and unboxing functionality for dpnp.ndarray to numba_dpex (#902)
* New DpexTarget and dispatcher for compiling dpnp using numba-dpex (#887)
* Overload implementation for dpnp.empty (#902)
* Overload implementation for dpnp.empty_like, dpnp.zeros_like and
  dpnp.ones_like inside dpjit (#928)
* Overload implementation for dpnp.zeros and dpnp.ones inside dpjit (#923)
* Compilation and offload support for dpnp vector style expressions using Numba
  parfors (#957)
* Compilation of over 70 ufuncs for dpnp inside dpjit (#957)
* Backported the split parfor pass from upstream Numba. (#949)
* Numba type aliases to numba_dpex. (#851)
* Numba prange alias inside numba_dpex. (#957)
* New LRU cache for kernels (#804) and funcs (#877)
* New Range and NdRange classes for kernel submission that follow sycl's range
  and ndrange syntax. (#888)
* Monkey pacthes to Numba 0.56.4 to support dpnp ufuncs, allocating dpnp
  arrays (#954)
* New config flag (NUMBA_DPEX_DUMP_KERNEL_LLVM) to dump a kernel's
  LLVM IR (#924)
* A badge to our gitter chatroom (#919)
* A small script to update copyright headers (#917)
* A new dpexrt_python extension to support USM allocators for Numba
  NRT_MemInfo (#902)
* Updated examples for kernel API demonstrating compute-follows-data programming
  model. (#826)

### Changed
* `CLK_GLOBAL_MEM_FENCE` and `CLK_LOCAL_MEM_FENCE` flags renamed to
  `GLOBAL_MEM_FENCE` and `LOCAL_MEM_FENCE`. (#844)
* Switched from Ubuntu-latest to Ubuntu-20.04 for conda package build (#836)
* Rename USMNdArrayType to USMNdArray (#851)
* Changes to the Numba type to represent dpnp ndarray typess now renamed to
  DpnpNdarray (#880)
* Improved exceptions and user errors (#804)
* Updated internal API for kernel interface with improved support for
  `__sycl_usm_array_interface__` protocol (#804)
* Pin generated spirv version for kernels to 1.1 (#885)
* Rename DpexContext and DpexTypingContext to DpexKernelTarget and
  DpexKernelTypingContext (#887)
* Renamed existing dpnp overloads that used stubs to dpnp_stubs_impl.py (#953)
* Dpctl version requirement mismatch is now a warning and not an
  ImportError (#925)
* Update to versioneer 0.28 (#827)
* Update to dpctl 0.14 (#858)
* Update linters: black to 23.1.0, isort to 5.12.0 (#900)
* License in setup.py to match actual project licensing (#904)

### Fixed
* Kernel specialization, compute follows data programming model for
  kernels (#804)
* Dispatcher/caching rewrite to address performance regression (#912, #896)
* func decorator qualname ambiguation fix (#905)

### Removed
* Removes the numpy_usm_shared module from numba_dpex. (#841)
* Removes the usage of llvmlite.llvmpy (#932)

### Deprecated
* Support for NumPy arrays as kernel arguments (#804)
* Kernel argument access specifiers (#804)
* Support for dpctl.device_context to launch kernels and njit offloading (#804)
* Dpnp overloads using stubs. (#953)

## [0.19.0] - 2022-11-21

### Added
* Supported numba0.56. (#818)
* Supported dpnp0.11 and dpctl0.14.
* Added customized exception classes. (#798)

### Fixed
* Fixed a crash when calling take() for input array with non-integer values. (#771)
* Fixed pairwise_distance.py to run on machine with no FP64 support in HW.  (#806)

## [0.18.1] - 2022-08-06

### Added
* Implemented support for `dpnp.empty()` (#728)

### Changed
* numba-dppy package is now renamed to numba-dpex.

## [0.18.0] - 2022-02-22

### Added
* Run coverage in GitHub Actions and upload results to coveralls.io (#621)
* Change black to only allow 80 char lines. Reformat sources. (#631)
* Ignore formatting changes from git-blame. (#632)
* Add `numba_support.py` with `numba_version` (#656)
* Add skip_no_numba055 decorator (#662)
* Parameterize test for atomics (#661)
* Reuse decorator `skip_no_opencl_Xpu` to skip tests (#663)
* Add decorator to skip unsupported atomics (#664)
* Support arrays with `__sycl_usm_array_interface__` attribute (#629)
* Support memory allocation in private address space (#640)
* Move skips for opencl to helper (#665)
* Support dpctl 0.12 (#669)
* Implement compute-follows-data programming model [kernel API] (#598)
* Use filter_str to skip tests on missing devices (#672)
* Add check for DPNP and pin MKL version in workflow and dev environment (#648)
* Add CODEOWNERS for distributing review process (#670)
* Add `skip_no_dpnp` and apply it to all tests (#668)
* Test skipping improvements (#675)
* Use Python 3.9 in dev environment and pin DPNP (#644)
* Add examples into package (#680)
* Make possible to force debugging tests (#681)
* Refactoring for debugging tests (#682)
* Adopt Numba 0.55 debugging features (#654)
* Run public CI on pull request (#695)
* Support for coverage in internal CI (#708)
* Update to dpnp 0.10 (#709)
* Update recipe with dpctl and dpnp version for build (#710)

### Changed
* Move `dpcpp/llvm-spirv` from runtime to testing dependency (#659)

### Fixed
* Fix building with DPNP (#674)
* Fix public CI: opencl driver, channel priority, dpctl version (#691)
* Fix codestyle black (#696)
* Fix documentation generation (#697)
* Load dpctl lib on Linux using `libDPCTLSyclInterface.so.0` (#707)
* Fix search llvm-spirv if dpcpp compiler package is not installed (#703)
* Pin dpnp version in runtime to allow dev versions of dpnp (#712)

## [0.17.4] - 2021-12-02

### Changed
* Move `dpcpp/llvm-spirv` from runtime to testing dependency by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/659

## [0.17.3] - 2021-11-30

### Changed
* Use `llvm-spirv` from `dpcpp` compiler package by default [cherry picked from #649] (#651)

### Fixed
* Enable offloading for `numba.njit` in `dpctl.deveice_context` (#630)
* Fix upload conditions for main and release branches (#610)
* Fix DeprecationWarning when use `version.parse()` [cherry picked from #639] (#642)

## [0.17.2] - 2021-11-15

### Changed
* Use llvm-spirv from bin-llvm during build for Linux and Windows (#626, #627)

## [0.17.1] - 2021-11-10

### Changed
* Update clang to icx (#622)

## [0.17.0] - 2021-11-03

### Added
* Use Python 3.9 [public CI] by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/574
* Use `NUMBA_DPPY_DEBUG` for debugging GDB tests by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/578
* Preliminary support Numba 0.55 (master branch) by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/583
* Workflow for manually running tests using Numba PRs by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/586
* Add public CI trigger on tags by @1e-to in https://github.com/IntelPython/numba-dppy/pull/589
* Upload packages for `release*` branches by @1e-to in https://github.com/IntelPython/numba-dppy/pull/593
* Update to dpctl 0.11 by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/595
* Update to dpnp 0.9 by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/599
* Improve the documenatation landing page by @diptorupd in https://github.com/IntelPython/numba-dppy/pull/601
* Clean up README by @diptorupd in https://github.com/IntelPython/numba-dppy/pull/604

### Fixed
* Restrict dpctl to 0.10.* for release 0.16 by @1e-to in https://github.com/IntelPython/numba-dppy/pull/590
* Fix upload from release branch by @1e-to in https://github.com/IntelPython/numba-dppy/pull/596
* Unskip tests passing with dpnp 0.9.0rc1 by @PokhodenkoSA in https://github.com/IntelPython/numba-dppy/pull/606

## [0.16.1] - 2021-10-20

### Changed
* Fix dpctl to 0.10 (#590)
* Add Public CI trigger for tags (#589)

## [0.16.0] - 2021-09-28

### Added
- Improve build and infra scripts (#544)
- Add docs about local variables lifetime (#534)
- Public CI for Windows (#536, #558)
- Add info about tags in documentation (#543)
- Add code coverage configurations (#561)
- Add support pytest-cov and pytest-xdist (#562)
- Add documentation workflow (#547)
- Test numba and numba-dppy API with GDB (#566)
- Transform commands scripts for GDB to tests (#568)

### Changed
- Update dpnp 0.8 (#524)
- Fix passing strides array to DPNP dot and matmul (#565)
- Use older compiler for backwards compatibility (#549)
- Update conda recipe dependency for dpnp (#535)
- Update dpctl 0.10 (memcpy async) (#529)
- Change channels priority in public CI (#532)
- Added runtime dependency `llvm-spirv 11.*` (#523)
- Update test matrix in README (#560)
- Use dpctl 0.10* and dpnp 0.8* in development configuration (environment.yml)

### Fixed
- Update test and fix typo for atomics (#550)
- Delete unused file `run_test.sh`
- Fix Public CI for using development packages (#522)
- Removed redundant import in docs (#521)

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
