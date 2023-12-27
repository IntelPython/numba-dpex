"""Launcher package to provide the same way of calling kernel as experimental
one."""


def call_kernel(kernel_fn, index_space, *kernel_args) -> None:
    """Syntax sugar for calling kernel the same way as experimental one.
    It is a temporary glue for the experimental kernel migration.

    Args:
        kernel_fn (numba_dpex.experimental.KernelDispatcher): A
        numba_dpex.kernel decorated function that is compiled to a
        KernelDispatcher by numba_dpex.
        index_space (Range | NdRange): A numba_dpex.Range or numba_dpex.NdRange
        type object that specifies the index space for the kernel.
        kernel_args : List of objects that are passed to the numba_dpex.kernel
        decorated function.
    """
    kernel_fn[index_space](*kernel_args)
