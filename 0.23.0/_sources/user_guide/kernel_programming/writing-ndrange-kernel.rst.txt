
In a range kernel, the kernel execution is scheduled over a set of work-items
without any explicit grouping of the work-items. The basic form of parallelism
that can be expressed using a range kernel does not allow expressing any notion
of locality within the kernel. To get around that limitation, kapi provides a
second form of expressing a parallel kernel that is called an *nd-range* kernel.
An nd-range kernel represents a data-parallel execution of the kernel by a set
of explicitly defined groups of work-items. An individual group of work-items is
called a *work-group*. :ref:`ex_matmul_kernel` demonstrates an nd-range kernel
and some of the advanced features programmers can use in this type of kernel.

.. code-block:: python
    :linenos:
    :caption: **Example:** Sliding window matrix multiplication as an nd-range kernel
    :name: ex_matmul_kernel

    from numba_dpex import kernel_api as kapi
    import numba_dpex as dpex
    import numpy as np
    import dpctl.tensor as dpt

    square_block_side = 2
    work_group_size = (square_block_side, square_block_side)
    dtype = np.float32


    @dpex.kernel
    def matmul(
        nditem: kapi.NdItem,
        X,  # IN READ-ONLY    (X_n_rows, n_cols)
        y,  # IN READ-ONLY    (n_cols, y_n_rows),
        X_slm,  # SLM to store a sliding window over X
        Y_slm,  # SLM to store a sliding window over Y
        result,  # OUT        (X_n_rows, y_n_rows)
    ):
        X_n_rows = X.shape[0]
        Y_n_cols = y.shape[1]
        n_cols = X.shape[1]

        result_row_idx = nditem.get_global_id(0)
        result_col_idx = nditem.get_global_id(1)

        local_row_idx = nditem.get_local_id(0)
        local_col_idx = nditem.get_local_id(1)

        n_blocks_for_cols = n_cols // square_block_side
        if (n_cols % square_block_side) > 0:
            n_blocks_for_cols += 1

        output = dtype(0)

        gr = nditem.get_group()

        for block_idx in range(n_blocks_for_cols):
            X_slm[local_row_idx, local_col_idx] = dtype(0)
            Y_slm[local_row_idx, local_col_idx] = dtype(0)
            if (result_row_idx < X_n_rows) and (
                (local_col_idx + (square_block_side * block_idx)) < n_cols
            ):
                X_slm[local_row_idx, local_col_idx] = X[
                    result_row_idx, local_col_idx + (square_block_side * block_idx)
                ]

            if (result_col_idx < Y_n_cols) and (
                (local_row_idx + (square_block_side * block_idx)) < n_cols
            ):
                Y_slm[local_row_idx, local_col_idx] = y[
                    local_row_idx + (square_block_side * block_idx), result_col_idx
                ]

            kapi.group_barrier(gr)

            for idx in range(square_block_side):
                output += X_slm[local_row_idx, idx] * Y_slm[idx, local_col_idx]

            kapi.group_barrier(gr)

        if (result_row_idx < X_n_rows) and (result_col_idx < Y_n_cols):
            result[result_row_idx, result_col_idx] = output


    def _arange_reshaped(shape, dtype):
        n_items = shape[0] * shape[1]
        return np.arange(n_items, dtype=dtype).reshape(shape)


    X = _arange_reshaped((5, 5), dtype)
    Y = _arange_reshaped((5, 5), dtype)
    X = dpt.asarray(X)
    Y = dpt.asarray(Y)
    device = X.device.sycl_device
    result = dpt.zeros((5, 5), dtype, device=device)
    X_slm = kapi.LocalAccessor(shape=work_group_size, dtype=dtype)
    Y_slm = kapi.LocalAccessor(shape=work_group_size, dtype=dtype)

    dpex.call_kernel(matmul, kapi.NdRange((6, 6), (2, 2)), X, Y, X_slm, Y_slm, result)


When writing an nd-range kernel, a programmer defines a set of groups of
work-items instead of a flat execution range.There are several semantic rules
associated both with a work-group and the work-items in a work-group:

- Each work-group gets executed in an arbitrary order by the underlying
  runtime and programmers should not assume any implicit ordering.

- Work-items in different wok-groups cannot communicate with each other except
  via atomic operations on global memory.

- Work-items within a work-group share a common memory region called
  "shared local memory" (SLM). Depending on the device the SLM maybe mapped to a
  dedicated fast memory.

- Work-items in a work-group can synchronize using a
  :func:`numba_dpex.kernel_api.group_barrier` operation that can additionally
  guarantee memory consistency using a *work-group memory fence*.

.. note::

    The SYCL language provides additional features for work-items in a
    work-group such as *group functions* that specify communication routines
    across work-items and also implement patterns such as reduction and scan.
    These features are not yet available in numba-dpex.

An nd-range kernel needs to be launched with an instance of the
:py:class:`numba_dpex.kernel_api.NdRange` class and the first
argument to an nd-range kernel has to be an instance of
:py:class:`numba_dpex.kernel_api.NdItem`. Apart from the need to provide an
```NdItem`` parameter, the rest of the semantic rules that apply to a range
kernel also apply to an nd-range kernel.
