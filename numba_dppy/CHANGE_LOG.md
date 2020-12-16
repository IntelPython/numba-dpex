# Changelog
All notable changes to this project will be documented in this file.

## [0.12.0] - 2020-12-17
### Added
- Dependency on llvm-spirv, llvmdev , spirv-tools.
- Documentation on using gdb.
- Implementation of `np.linalg.eig`.
- Offload diagnostics.

### Changed
- Rename `DPPL` to `DPPY`.
- `DPPLTestCase` replaced with `unittest.TestCase`.
- Numba-dppy tests and examples use `with device_context`.

### Removed
- Use of cffi.
- The old backup file.

## NUMBA Version 0.48.0 + DPPY Version 0.3.0 (June 29, 2020)

This release includes:
- Caching of dppy.kernels which will improve performance.
- Addition of support for Intel Advisor which will help in profiling applications.
