## Debugging with GDB

Setting the debug environment variable `NUMBA_DPPY_DEBUG` (e.g. `export NUMBA_DPPY_DEBUG=True`) enables 
the emission of debug info to the llvm and spirv IR.
To disable debugging set this variable to None: (e.g. `export NUMBA_DPPL_DEBUG= `).  
Currently, the following debug info is available:
- Source location (filename and line number) is available. 
- Setting break points by the line number.
- Stepping over break points.

### Requirements

Intel® Distribution for GDB installed to the system.  
Documentation for this debugger can be found in the 
[Intel® Distribution for GDB documentation](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-gdb.html).

### Example debug usage

```bash
$ export NUMBA_DPPY_DEBUG=True
$ gdb-oneapi -q python  
(gdb) break numba_dppy/examples/sum.py:14     # Assumes the kernel is in file sum.py, at line 14  
(gdb) run sum.py
```

### Limitations

Currently, Numba-dppy provides only initial support of debugging GPU kernels.  
The following functionality is **not supported** :
- Printing kernel local variables (e.g. ```info locals```).
- Stepping over several off-loaded functions.
