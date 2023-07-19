.. include:: ./../../ext_links.txt

Debugging Local Variables
=========================

Several conditions could influence debugging of local variables:

1. :ref:`numba-opt`
2. :ref:`local-variables-lifetime`

.. _numba-opt:

Optimization Level for LLVM
---------------------------

Numba provides environment variable ``NUMBA_OPT`` for configuring
optimization level for LLVM. The default optimization level is three.
Refer `Numba documentation`_ for details. It is recommended to debug with
:samp:`NUMBA_OPT=0`. The possible effect of various optimization levels may be
as follows:

* :samp:`NUMBA_OPT=0` means "no optimization" level - all local variables are available.
* :samp:`NUMBA_OPT=1` or higher levels - some variables may be optimized out.

Example
```````

Source code :file:`numba_dpex/examples/debug/sum_local_vars.py`:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/sum_local_vars.py
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
    (gdb) run numba_dpex/examples/debug/sum_local_vars.py
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
    (gdb) run numba_dpex/examples/debug/sum_local_vars.py
    ...
    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], ?? () at sum_local_vars.py:26 from /tmp/kernel_11059955544143858990_e6df1e.dbgelf
    26          c[i] = l1 + l2
    (gdb) info locals
    No locals.

It optimized out local variables ``i``, ``l1`` and ``l2`` with this optimization level.

.. _local-variables-lifetime:

Local Variables Lifetime in Numba IR
------------------------------------

Lifetime of Python variables are different from lifetime of variables in compiled code.
Numba analyses variables lifetime and try to optimize it.
The debugger can show the variable values, but they may be zeros
after the variable is explicitly deleted when the scope of variable is ended.

See `Numba variable policy <https://numba.pydata.org/numba-doc/latest/developer/live_variable_analysis.html?highlight=delete#live-variable-analysis>`_.

Numba provides environment variable ``NUMBA_EXTEND_VARIABLE_LIFETIMES``
for extending the lifetime of variables to the end of the block in which their lifetime ends.

See `Numba documentation`_.

Default is zero.

It is recommended to debug with :samp:`NUMBA_EXTEND_VARIABLE_LIFETIMES=1`.

Example 1 - Using ``NUMBA_EXTEND_VARIABLE_LIFETIMES``
`````````````````````````````````````````````````````

Source code :file:`numba_dpex/tests/debugging/test_info.py`:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/side-by-side.py
   :pyobject: common_loop_body
   :linenos:
   :lineno-match:
   :emphasize-lines: 5

Debug session with :samp:`NUMBA_EXTEND_VARIABLE_LIFETIMES=1`:

.. code-block:: shell-session
    :emphasize-lines: 3, 10-12

    $ gdb-oneapi -q python
    ...
    (gdb) set environment NUMBA_EXTEND_VARIABLE_LIFETIMES 1
    (gdb) break side-by-side.py:28
    ...
    (gdb) run numba_dpex/examples/debug/side-by-side.py --api=numba-dpex-kernel
    ...
    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::common_loop_body (param_a=0, param_b=0) at side-by-side.py:28
    28          return result
    (gdb) info locals
    param_c = 10
    param_d = 0
    result = 10

It prints values of ``param_c`` and ``param_d``.

Debug session with :samp:`NUMBA_EXTEND_VARIABLE_LIFETIMES=0`:

.. code-block:: shell-session
    :emphasize-lines: 3, 10-12

    $ gdb-oneapi -q python
    ...
    (gdb) set environment NUMBA_EXTEND_VARIABLE_LIFETIMES 0
    (gdb) break side-by-side.py:28
    ...
    (gdb) run numba_dpex/examples/debug/side-by-side.py --api=numba-dpex-kernel
    ...
    Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], __main__::common_loop_body (param_a=0, param_b=0) at side-by-side.py:28
    28          return result
    (gdb) info locals
    param_c = 0
    param_d = 0
    result = 10

.. _example-NUMBA_DUMP_ANNOTATION:

Example 2 - Using ``NUMBA_DUMP_ANNOTATION``
```````````````````````````````````````````

Source code :file:`numba_dpex/examples/debug/sum_local_vars.py`:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/sum_local_vars.py
    :pyobject: data_parallel_sum
    :linenos:
    :lineno-match:

Run this code with environment variable :samp:`NUMBA_DUMP_ANNOTATION=1` and it
will show where numba inserts `del` for variables.

.. code-block::
    :linenos:
    :emphasize-lines: 28

    -----------------------------------ANNOTATION-----------------------------------
    # File: numba_dpex/examples/debug/sum_local_vars.py
    # --- LINE 20 ---

    @numba_dpex.kernel(debug=True)

    # --- LINE 21 ---

    def data_parallel_sum(a, b, c):

        # --- LINE 22 ---
        # label 0
        #   a = arg(0, name=a)  :: array(float32, 1d, C)
        #   b = arg(1, name=b)  :: array(float32, 1d, C)
        #   c = arg(2, name=c)  :: array(float32, 1d, C)
        #   $2load_global.0 = global(dpex: <module 'numba_dpex' from '.../numba-dpex/numba_dpex/__init__.py'>)  :: Module(<module 'numba_dpex' from '.../numba-dpex/numba_dpex/__init__.py'>)
        #   $4load_method.1 = getattr(value=$2load_global.0, attr=get_global_id)  :: Function(<function get_global_id at 0x7f82b8bae430>)
        #   del $2load_global.0
        #   $const6.2 = const(int, 0)  :: Literal[int](0)
        #   i = call $4load_method.1($const6.2, func=$4load_method.1, args=[Var($const6.2, sum_local_vars.py:22)], kws=(), vararg=None, target=None)  :: (uint32,) -> int64
        #   del $const6.2
        #   del $4load_method.1

        i = dpex.get_global_id(0)

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

.. literalinclude:: ./../../../../numba_dpex/examples/debug/sum_local_vars_revive.py
    :lines: 5-
    :linenos:
    :lineno-match:

.. code-block::
    :linenos:
    :emphasize-lines: 59

    -----------------------------------ANNOTATION-----------------------------------
    # File: numba_dpex/examples/debug/sum_local_vars_revive.py
    # --- LINE 24 ---

    @numba_dpex.kernel(debug=True)

    # --- LINE 25 ---

    def data_parallel_sum(a, b, c):

        # --- LINE 26 ---
        # label 0
        #   a = arg(0, name=a)  :: array(float32, 1d, C)
        #   b = arg(1, name=b)  :: array(float32, 1d, C)
        #   c = arg(2, name=c)  :: array(float32, 1d, C)
        #   $2load_global.0 = global(dpex: <module 'numba_dpex' from '.../numba-dpex/numba_dpex/__init__.py'>)  :: Module(<module 'numba_dpex' from '.../numba-dpex/numba_dpex/__init__.py'>)
        #   $4load_method.1 = getattr(value=$2load_global.0, attr=get_global_id)  :: Function(<function get_global_id at 0x7fcdf7e8c4c0>)
        #   del $2load_global.0
        #   $const6.2 = const(int, 0)  :: Literal[int](0)
        #   i = call $4load_method.1($const6.2, func=$4load_method.1, args=[Var($const6.2, sum_local_vars_revive.py:26)], kws=(), vararg=None, target=None)  :: (uint32,) -> int64
        #   del $const6.2
        #   del $4load_method.1

        i = dpex.get_global_id(0)

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
        #   $48load_global.19 = global(revive: <numba_dpex.compiler.DpexFunctionTemplate object at 0x7fce12e5cc40>)  :: Function(<numba_dpex.compiler.DpexFunctionTemplate object at 0x7fce12e5cc40>)
        #   $52call_function.21 = call $48load_global.19(a, func=$48load_global.19, args=[Var(a, sum_local_vars_revive.py:26)], kws=(), vararg=None, target=None)  :: (array(float32, 1d, C),) -> array(float32, 1d, C)
        #   del a
        #   del $52call_function.21
        #   del $48load_global.19
        #   $const56.22 = const(NoneType, None)  :: none
        #   $58return_value.23 = cast(value=$const56.22)  :: none
        #   del $const56.22
        #   return $58return_value.23

        revive(a)  # pass variable to dummy function

Run with environment variables :samp:`NUMBA_DUMP_ANNOTATION=1` and
:samp:`NUMBA_EXTEND_VARIABLE_LIFETIMES=1`.
It will show that numba inserts `del` for variables at the end of the block:

.. code-block::
    :linenos:
    :emphasize-lines: 11-25

    -----------------------------------ANNOTATION-----------------------------------
    # File: numba_dpex/examples/debug/sum_local_vars.py
    ...
    def data_parallel_sum(a, b, c):
        ...
        # --- LINE 26 ---
        #   $40binary_add.16 = l1 + l2  :: float64
        #   c[i] = $40binary_add.16  :: (array(float32, 1d, C), int64, float64) -> none
        #   $const48.19 = const(NoneType, None)  :: none
        #   $50return_value.20 = cast(value=$const48.19)  :: none
        #   del $2load_global.0
        #   del $const6.2
        #   del $4load_method.1
        #   del a
        #   del $const18.7
        #   del $16binary_subscr.6
        #   del b
        #   del $const30.12
        #   del $28binary_subscr.11
        #   del l2
        #   del l1
        #   del i
        #   del c
        #   del $40binary_add.16
        #   del $const48.19
        #   return $50return_value.20

        c[i] = l1 + l2


Example 3 - Using ``info locals``
`````````````````````````````````

Source code :file:`sum_local_vars.py`:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/sum_local_vars.py
    :lines: 5-
    :linenos:
    :lineno-match:

Run the debugger with ``NUMBA_OPT=0``:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 1-6

Run the ``info locals`` command. The sample output on "no optimization" level ``NUMBA_OPT=0`` is as follows:

.. literalinclude:: ./../../../../numba_dpex/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 8-48
    :emphasize-lines: 1-16, 24-39

Since the debugger does not hit a line with the target variable ``l1``, the value equals 0. The true value of the variable ``l1`` is shown after stepping to line 22.

.. literalinclude:: ./../../../../numba_dpex/examples/debug/commands/docs/local_variables_0
    :language: shell-session
    :lines: 49-66
    :emphasize-lines: 1-16

When the debugger hits the last line of the kernel, ``info locals`` command returns all the local variables with their values.

.. _`Numba documentation`: https://numba.readthedocs.io/en/stable/reference/envvars.html?#envvar-NUMBA_OPT
