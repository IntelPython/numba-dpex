Debugging the compilation pipeline
==================================

Consider the following two examples. ``numba_dppy/examples/debug/simple_sum.py``:

.. literalinclude:: ../../numba_dppy/examples/debug/simple_sum.py
    :lines: 15-
    :linenos:

Example with njit:

.. literalinclude:: ../../numba_dppy/examples/debug/njit_basic.py
    :lines: 15-
    :linenos:

Getting the DWARF from .elf file
--------------------------------

If you need to get the DWARF information from a specific kernel, just enable ``IGC_ShaderDumpEnable`` variable.
IGC will write number of dumps into ``/tmp/IntelIGC``.

.. code-block:: bash

    export IGC_ShaderDumpEnable=1

To read the DWARF of a kernel, we first need a copy of the IGC generated kernel binary.
To do that, run the Python script in a debugger, and set a breakpoint in the kernel:

.. code-block:: bash

    export NUMBA_DPPY_DEBUGINFO=1
    export NUMBA_OPT=1
    gdb-oneapi -q --args python simple_sum.py
    gdb-oneapi -q python
    (gdb) break simple_sum.py:22     # Assumes the kernel is in file simple_sum.py, at line 22
    (gdb) run

Once the breakpoint is hit, the kernel has been generated and offloaded.
At that point, the IGFX driver (i.e. our debugger driver) has copied the kernel into a file, and saved it at ``/tmp``.
All files saved at ``/tmp/IntelIGC/python_xxx/``

Dump generated DWARF from the kernel binary (elf) via llvm-dwarfdump tool:

.. code-block:: bash

    llvm-dwarfdump xxx.elf

Getting the DWARF from Numba assembly (for njit)
--------------------------------------------------

Setting Numba environment variable ``NUMBA_DUMP_ASSEMBLY`` dumps the native assembly code of compiled functions.

.. code-block:: bash

    NUMBA_DUMP_ASSEMBLY=1 python njit-basic.py > njit-basic-asm.txt

Clear file from prints and unrecognized characters (-, =, !, )
Get compiled from assembler:

.. code-block:: bash

    as -o o njit-basic-asm.txt

Get dwarf with objdump:

.. code-block:: bash

    objdump -W o > o_dwarf

See also:

    - `NUMBA_DUMP_ASSEMBLY <https://numba.pydata.org/numba-doc/dev/reference/envvars.html#envvar-NUMBA_DUMP_ASSEMBLY>`_
