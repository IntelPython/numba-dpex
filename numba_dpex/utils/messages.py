# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
