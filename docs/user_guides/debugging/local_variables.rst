Debugging Local Variables
=========================

Several conditions could influence debugging of local variables:

1. :ref:`numba-opt`
2. :ref:`local-variables-lifetime`

.. _numba-opt:

Optimization Level for LLVM
---------------------------

Numba provides environment variable :ref:`NUMBA_OPT` for configuring optimization level for LLVM.

See `Numba documentation <https://numba.readthedocs.io/en/stable/reference/envvars.html?#envvar-NUMBA_OPT>`_.

* :samp:`NUMBA_OPT=0` means "no optimization" level - all local variables are available.
* :samp:`NUMBA_OPT=1` or higher levels - some variables may be optimized out.

Default value is 3.

It is recommended to debug with :samp:`NUMBA_OPT=0`.

Example
```````

Source code :file:`numba_dppy/examples/debug/sum_local_vars.py`:

.. literalinclude:: ../../../numba_dppy/examples/debug/sum_local_vars.py
    :pyobject: data_parallel_sum
    :linenos:
    :lineno-match:
    :emphasize-lines: 6

Debug session with :samp:`NUMBA_OPT=0`:

.. code-block:: shell-session
    :emphasize-lines: 3, 10-13

    $ gdb-oneapi -q python
    ...
    (gdb) set environment NUMBA_OPT 0
    (gdb) break sum_local_vars.py:26
    ...
    (gdb) run numba_dppy/examples/debug/sum_local_vars.py
    ...
    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::data_parallel_sum (a=..., b=..., c=...) at sum_local_vars.py:26
    26          c[i] = l1 + l2
    (gdb) info locals
    i = 0
    l1 = 2.9795852899551392
    l2 = 0.22986688613891601

It printed all local variables with their values.

Debug session with :samp:`NUMBA_OPT=1`:

.. code-block:: shell-session
    :emphasize-lines: 3, 10-11

    $ gdb-oneapi -q python
    ...
    (gdb) set environment NUMBA_OPT 1
    (gdb) break sum_local_vars.py:26
    ...
    (gdb) run numba_dppy/examples/debug/sum_local_vars.py
    ...
    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], ?? () at sum_local_vars.py:26 from /tmp/kernel_11059955544143858990_e6df1e.dbgelf
    26          c[i] = l1 + l2
    (gdb) info locals
    No locals.

It optimized out local variables ``i``, ``l1`` and ``l2`` with this optimization level.

``print <variable>``
------------------

To print the value of a variable, run the ``print <variable>`` command.

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 67-72
    :emphasize-lines: 1-6

.. note::

    Kernel variables are shown in intermidiate representation view (with "$" sign). The actual values of the arrays are currently not available.

``ptype <variable>``
------------------

To print the type of a variable, run the ``ptype <variable>`` or ``whatis <variable>`` commands:

.. literalinclude:: ../../../numba_dppy/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 73-81
    :emphasize-lines: 1-6

See also:

    - `Local variables in GDB* <https://sourceware.org/gdb/current/onlinedocs/gdb/Frame-Info.html#Frame-Info>`_

.. _local-variables-lifetime:

Lifetime of local variables
---------------------------

Numba uses live variable analysis.
Lifetime of Python variables are different from lifetime of variables in
compiled code.

.. note::
    For more information, refer to `Numba variable policy <https://numba.pydata.org/numba-doc/latest/developer/live_variable_analysis.html?highlight=delete#live-variable-analysis>`_.



It affects debugging experience in following way.

Consider Numba-dppy kernel code from :file:`sum_local_vars.py`:

.. literalinclude:: ../../../numba_dppy/examples/debug/sum_local_vars.py
    :lines: 20-25
    :linenos:
    :lineno-match:

Run this code with environment variable :samp:`NUMBA_DUMP_ANNOTATION=1` and it
will show where numba inserts `del` for variables.

.. code-block::
    :linenos:
    :emphasize-lines: 28

    -----------------------------------ANNOTATION-----------------------------------
    # File: numba_dppy/examples/debug/sum_local_vars.py
    # --- LINE 20 ---

    @dppy.kernel(debug=True)

    # --- LINE 21 ---

    def data_parallel_sum(a, b, c):

        # --- LINE 22 ---
        # label 0
        #   a = arg(0, name=a)  :: array(float32, 1d, C)
        #   b = arg(1, name=b)  :: array(float32, 1d, C)
        #   c = arg(2, name=c)  :: array(float32, 1d, C)
        #   $2load_global.0 = global(dppy: <module 'numba_dppy' from '.../numba-dppy/numba_dppy/__init__.py'>)  :: Module(<module 'numba_dppy' from '.../numba-dppy/numba_dppy/__init__.py'>)
        #   $4load_method.1 = getattr(value=$2load_global.0, attr=get_global_id)  :: Function(<function get_global_id at 0x7f82b8bae430>)
        #   del $2load_global.0
        #   $const6.2 = const(int, 0)  :: Literal[int](0)
        #   i = call $4load_method.1($const6.2, func=$4load_method.1, args=[Var($const6.2, sum_local_vars.py:22)], kws=(), vararg=None, target=None)  :: (uint32,) -> int64
        #   del $const6.2
        #   del $4load_method.1

        i = dppy.get_global_id(0)

        # --- LINE 23 ---
        #   $16binary_subscr.6 = getitem(value=a, index=i, fn=<built-in function getitem>)  :: float32
        #   del a
        #   $const18.7 = const(float, 2.5)  :: float64
        #   l1 = $16binary_subscr.6 + $const18.7  :: float64
        #   del $const18.7
        #   del $16binary_subscr.6

        l1 = a[i] + 2.5

        # --- LINE 24 ---
        #   $28binary_subscr.11 = getitem(value=b, index=i, fn=<built-in function getitem>)  :: float32
        #   del b
        #   $const30.12 = const(float, 0.3)  :: float64
        #   l2 = $28binary_subscr.11 * $const30.12  :: float64
        #   del $const30.12
        #   del $28binary_subscr.11

        l2 = b[i] * 0.3

        # --- LINE 25 ---
        #   $40binary_add.16 = l1 + l2  :: float64
        #   del l2
        #   del l1
        #   c[i] = $40binary_add.16  :: (array(float32, 1d, C), int64, float64) -> none
        #   del i
        #   del c
        #   del $40binary_add.16
        #   $const48.19 = const(NoneType, None)  :: none
        #   $50return_value.20 = cast(value=$const48.19)  :: none
        #   del $const48.19
        #   return $50return_value.20

        c[i] = l1 + l2

I.e. in `LINE 23` variable `a` used the last time and numba inserts `del a` as
shown in annotated code in line 28. It means you will see value 0 for the
variable `a` when you set breakpoint at `LINE 24`.

As a workaround you can expand lifetime of the variable by using it (i.e.
passing to dummy function `revive()`) at the end of the function. So numba will
not insert `del a` until the end of the function.

.. literalinclude:: ../../../numba_dppy/examples/debug/sum_local_vars_revive.py
    :lines: 20-31
    :linenos:
    :lineno-match:

.. code-block::
    :linenos:
    :emphasize-lines: 59

    -----------------------------------ANNOTATION-----------------------------------
    # File: numba_dppy/examples/debug/sum_local_vars_revive.py
    # --- LINE 24 ---

    @dppy.kernel(debug=True)

    # --- LINE 25 ---

    def data_parallel_sum(a, b, c):

        # --- LINE 26 ---
        # label 0
        #   a = arg(0, name=a)  :: array(float32, 1d, C)
        #   b = arg(1, name=b)  :: array(float32, 1d, C)
        #   c = arg(2, name=c)  :: array(float32, 1d, C)
        #   $2load_global.0 = global(dppy: <module 'numba_dppy' from '.../numba-dppy/numba_dppy/__init__.py'>)  :: Module(<module 'numba_dppy' from '.../numba-dppy/numba_dppy/__init__.py'>)
        #   $4load_method.1 = getattr(value=$2load_global.0, attr=get_global_id)  :: Function(<function get_global_id at 0x7fcdf7e8c4c0>)
        #   del $2load_global.0
        #   $const6.2 = const(int, 0)  :: Literal[int](0)
        #   i = call $4load_method.1($const6.2, func=$4load_method.1, args=[Var($const6.2, sum_local_vars_revive.py:26)], kws=(), vararg=None, target=None)  :: (uint32,) -> int64
        #   del $const6.2
        #   del $4load_method.1

        i = dppy.get_global_id(0)

        # --- LINE 27 ---
        #   $16binary_subscr.6 = getitem(value=a, index=i, fn=<built-in function getitem>)  :: float32
        #   $const18.7 = const(float, 2.5)  :: float64
        #   l1 = $16binary_subscr.6 + $const18.7  :: float64
        #   del $const18.7
        #   del $16binary_subscr.6

        l1 = a[i] + 2.5

        # --- LINE 28 ---
        #   $28binary_subscr.11 = getitem(value=b, index=i, fn=<built-in function getitem>)  :: float32
        #   del b
        #   $const30.12 = const(float, 0.3)  :: float64
        #   l2 = $28binary_subscr.11 * $const30.12  :: float64
        #   del $const30.12
        #   del $28binary_subscr.11

        l2 = b[i] * 0.3

        # --- LINE 29 ---
        #   $40binary_add.16 = l1 + l2  :: float64
        #   del l2
        #   del l1
        #   c[i] = $40binary_add.16  :: (array(float32, 1d, C), int64, float64) -> none
        #   del i
        #   del c
        #   del $40binary_add.16

        c[i] = l1 + l2

        # --- LINE 30 ---
        #   $48load_global.19 = global(revive: <numba_dppy.compiler.DPPYFunctionTemplate object at 0x7fce12e5cc40>)  :: Function(<numba_dppy.compiler.DPPYFunctionTemplate object at 0x7fce12e5cc40>)
        #   $52call_function.21 = call $48load_global.19(a, func=$48load_global.19, args=[Var(a, sum_local_vars_revive.py:26)], kws=(), vararg=None, target=None)  :: (array(float32, 1d, C),) -> array(float32, 1d, C)
        #   del a
        #   del $52call_function.21
        #   del $48load_global.19
        #   $const56.22 = const(NoneType, None)  :: none
        #   $58return_value.23 = cast(value=$const56.22)  :: none
        #   del $const56.22
        #   return $58return_value.23

        revive(a)  # pass variable to dummy function
