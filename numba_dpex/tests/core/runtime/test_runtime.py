# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba_dpex.core.runtime import _dpexrt_python


def test_pointers_exposed():
    """This test is to check attributts in _dpexrt_python."""

    def exposed(function_name):
        assert hasattr(_dpexrt_python, function_name)
        assert isinstance(getattr(_dpexrt_python, function_name), int)

    exposed("DPEXRT_sycl_usm_ndarray_from_python")
    exposed("DPEXRT_sycl_usm_ndarray_to_python_acqref")
    exposed("DPEXRT_MemInfo_alloc")
    exposed("NRT_ExternalAllocator_new_for_usm")
