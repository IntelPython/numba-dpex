# numba-dppy

Below is the functionality that is implemented in numba-dppy. You can follow the detailed descriptions of some of the features.

## Offload Diagnostics

Setting the debug environment variable `NUMBA_DPPY_OFFLOAD_DIAGNOSTICS `
(e.g. `export NUMBA_DPPY_OFFLOAD_DIAGNOSTICS=1`) enables the parallel and offload diagnostics information.

If set to an integer value between 1 and 4 (inclusive) diagnostic information about parallel transforms undertaken by Numba will be written to STDOUT. The higher the value set the more detailed the information produced.
In the "Auto-offloading" section there is the information on which device (device name) this parfor or kernel was offloaded.

## Controllable Fallback

With the default behavior of numba-dppy, if a section of code cannot be offloaded on the GPU, then it is automatically executed on the CPU and printed a warning. This behavior only applies to njit functions and auto-offloading of numpy functions, array expressions, and prange loops.

Setting the debug environment variable `NUMBA_DPPY_FALLBACK_OPTION `
(e.g. `export NUMBA_DPPY_FALLBACK_OPTION=0`) enables the code is not automatically offload to the CPU, and an error occurs. This is necessary in order to understand at an early stage which parts of the code do not work on the GPU, and not to wait for the program to execute on the CPU if you don't need it.
