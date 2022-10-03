# SPDX-FileCopyrightText: 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

"""The module stores a set of exception classes specific to numba_dpex compiler
pipeline.
"""


class KernelHasReturnValueError(Exception):
    """Exception raised when a kernel function is defined with a return
    statement.


    @numba_dpex.kernel does not allow users to return any value. The
    restriction is inline with the general ABI for device functions in OpenCL,
    CUDA, and SYCL.

    Args:
        return_type: Numba type representing the return value specified for
        the kernel function.

    """

    def __init__(self, return_type) -> None:
        self.return_type = return_type
        self.message = (
            f"Encountered a return value of type {self.return_type}. "
            f"A numba-dpex kernel must have a void return value."
        )

        super().__init__(self.message)
