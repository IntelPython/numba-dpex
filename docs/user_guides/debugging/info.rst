Info commands
=============

``info functions`` command
--------------------------

Investigating **info func** command on `CPU` in `jit` code.

Displays the list of functions in the debugged program.

**Syntax**
``````````

.. code-block::

    info functions
    info functions [Regex]

Parameters
``````````

`Regex`

If specified, the info functions command lists the functions matching the regex.
If omitted, the command lists all functions in all loaded modules (main program and shared libraries).

.. note::

    Running the ``info functions`` command without arguments may produce a lot of output
    as the list of all functions in all loaded shared libraries is typically very long.

Example
```````

Consider Python script :file:`njit-basic.py`:

.. code-block:: python
    :linenos:

    from numba import njit

    @njit (debug=True)
    def foo (arg):
        l1 = arg + 6        	# numba-kernel-breakpoint
        l2 = arg * 5.43             # second-line
        l3 = (arg, l1, l2, "bar")   # third-line
        print (arg, l1, l2, l3)     # fourth-line

    def main():
        result = foo(5)
        result1 = foo(5.3)

    if __name__ == '__main__':
        main()

Run debuger:

.. code-block:: bash

    NUMBA_OPT=0 gdb-oneapi -q python

Set breakpoint, run and see information about functions when the GDB debugger stops at the breakpoint:

.. code-block::

    (gdb) break njit-basic.py:8         # Set breakpoint in jit code
    (gdb) run njit-basic.py

    Breakpoint 1, __main__::foo$241 () at njit-basic.py:8
    8          print (arg, l1, l2, l3)     # fourth-line
    (gdb) info func foo
    All functions matching regular expression "foo":

    File njit-basic.py:
    4:      void __main__::foo$241(long long);

    Non-debugging symbols:
    0x00007ffff634f190  lzma_stream_footer_encode@plt
    0x00007ffff634f3c0  lzma_stream_footer_decode@plt
    0x00007ffff6358190  lzma_stream_footer_encode
    0x00007ffff635dd00  lzma_stream_footer_decode
    0x00007fffefd4dd90  mkl_sparse_d_void_foo@plt
    0x00007fffefd4f4f0  mkl_sparse_s_void_foo@plt

After :samp:`run njit-basic.py` command, we get the jit function compiled for the :obj:`result` variable.
We have only one compiled jit function at this stage, so :samp:`info func foo` will show one void function.

Use :samp:`с` to jump to the breakpoint of the jit function for the :obj:`result1` variable:

.. code-block::

    (gdb) c
    Continuing.
    5 11 27.15 (5, 11, 27.15, 'bar')

    Breakpoint 1, __main__::foo$242 () at njit-basic.py:8
    8           print (arg, l1, l2, l3)     # fourth-line
    (gdb) info func foo
    All functions matching regular expression "foo":

    File njit-basic.py:
    4:      void __main__::foo$241(long long);
    4:      void __main__::foo$242(double);

    Non-debugging symbols:
    0x00007ffff634f190  lzma_stream_footer_encode@plt
    0x00007ffff634f3c0  lzma_stream_footer_decode@plt
    0x00007ffff6358190  lzma_stream_footer_encode
    0x00007ffff635dd00  lzma_stream_footer_decode
    0x00007fffefd4dd90  mkl_sparse_d_void_foo@plt

We have two compiled jit functions at this stage, so :samp:`info func foo` will show two void functions.

Use Regex parameter to remove `Non-debugging symbols` output, e.g. :samp:`^__.*foo`:

.. code-block::

    (gdb) info func ^__.*foo
    All functions matching regular expression "^__.*foo":

    File njit-basic.py:
    4:      void __main__::foo$241(long long);
    4:      void __main__::foo$242(double);
