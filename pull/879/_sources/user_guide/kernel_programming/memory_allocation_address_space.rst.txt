Supported Address Space Qualifiers
==================================

The address space qualifier may be used to specify the region of memory that is
used to allocate the object.

Numba-dpex supports three disjoint named address spaces:

1. Global Address Space
    Global Address Space refers to memory objects allocated from the global
    memory pool and will be shared among all work-items. Arguments passed to any
    kernel are allocated in the global address space. In the below example,
    arguments `a`, `b` and `c` will be allocated in the global address space:

    .. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py


2. Local Address Space
    Local Address Space refers to memory objects that need to be allocated in
    local memory pool and are shared by all work-items of a work-group.
    Numba-dpex does not support passing arguments that are allocated in the
    local address space to `@numba_dpex.kernel`. Users are allowed to allocate
    static arrays in the local address space inside the `@numba_dpex.kernel`. In
    the example below `numba_dpex.local.array(shape, dtype)` is the API used to
    allocate a static array in the local address space:

    .. literalinclude:: ./../../../../numba_dpex/examples/barrier.py
      :lines: 54-87

3. Private Address Space
    Private Address Space refers to memory objects that are local to each
    work-item and is not shared with any other work-item. In the example below
    `numba_dpex.private.array(shape, dtype)` is the API used to allocate a
    static array in the private address space:

    .. literalinclude:: ./../../../../numba_dpex/examples/kernel/kernel_private_memory.py
