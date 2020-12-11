## Debugging with GDB

Setting the debug environment variable `NUMBA_DPPY_DEBUG` (e.g. `export NUMBA_DPPY_DEBUG=True`) enables the emission of debug info to 
the llvm and spirv IR. To disable debugging set this variable to None: (e.g. `export NUMBA_DPPL_DEBUG=`).  
Currently, the following debug info is available:
- Source location (filename and line number) is available. 
- Setting break points by the line number.
- Stepping over break points.

### Requirements
Intel GDB installed to the system  
follow the instruction: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-gdb.html

### Example debug usage
```bash
$ gdb -q python  
(gdb) break sum.py:13     # Assumes the kernel is in file sum.py, at line 13  
(gdb) run sum.py
```
