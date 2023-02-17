# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import llvmlite.binding as llb

from numba_dpex.core import runtime


def test_llvm_symbol_registered():
    """ "Register the helper function in _dpexrt_python so that we can insert calls to them via llvmlite.

    1. DPEXRT_sycl_usm_ndarray_from_python

    2.DPEXRT_sycl_usm_ndarray_to_python_acqref

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
