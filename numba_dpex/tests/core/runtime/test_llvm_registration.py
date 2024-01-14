# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import llvmlite.binding as llb

from numba_dpex.core import runtime


def test_llvm_symbol_registered():
    """Checks if the functions in the _dpexrt_python module are accessible
    using llvmlite.
    """
    assert (
        llb.address_of_symbol("DPEXRT_sycl_usm_ndarray_from_python")
        == runtime._dpexrt_python.DPEXRT_sycl_usm_ndarray_from_python
    )

    assert (
        llb.address_of_symbol("DPEXRT_sycl_usm_ndarray_to_python_acqref")
        == runtime._dpexrt_python.DPEXRT_sycl_usm_ndarray_to_python_acqref
    )

    assert (
        llb.address_of_symbol("NRT_ExternalAllocator_new_for_usm")
        == runtime._dpexrt_python.NRT_ExternalAllocator_new_for_usm
    )

    assert (
        llb.address_of_symbol("DPEXRT_sycl_queue_from_python")
        == runtime._dpexrt_python.DPEXRT_sycl_queue_from_python
    )

    assert (
        llb.address_of_symbol("DPEXRT_sycl_queue_to_python")
        == runtime._dpexrt_python.DPEXRT_sycl_queue_to_python
    )

    assert (
        llb.address_of_symbol("DPEXRTQueue_CreateFromFilterString")
        == runtime._dpexrt_python.DPEXRTQueue_CreateFromFilterString
    )

    assert (
        llb.address_of_symbol("DpexrtQueue_SubmitRange")
        == runtime._dpexrt_python.DpexrtQueue_SubmitRange
    )

    assert (
        llb.address_of_symbol("DPEXRT_MemInfo_alloc")
        == runtime._dpexrt_python.DPEXRT_MemInfo_alloc
    )

    assert (
        llb.address_of_symbol("DPEXRT_MemInfo_fill")
        == runtime._dpexrt_python.DPEXRT_MemInfo_fill
    )
