# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""This module provides content of warning and error messages."""

cfd_ctx_mgr_wrng_msg = (
    "Compute will follow data! Please do not use context manager "
    "to specify a SYCL queue to submit the kernel. The queue will be selected "
    "from the data."
)

IndeterminateExecutionQueueError_msg = (
    "Data passed as argument are not equivalent. Please "
    "create dpctl.tensor.usm_ndarray with equivalent SYCL queue."
)

mix_datatype_err_msg = (
    "Datatypes of array passed to @numba_dpex.kernel "
    "has to be the same. Passed datatypes: "
)
