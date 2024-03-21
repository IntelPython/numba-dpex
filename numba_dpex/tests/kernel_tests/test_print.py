# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex

list_of_dtypes = [
    dpnp.int32,
    dpnp.int64,
    dpnp.float32,
    dpnp.float64,
]

skip_on_gpu = pytest.mark.skipif(
    dpctl.SyclDevice().device_type == dpctl.device_type.gpu,
    reason="skip print on gpu",
)


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = dpnp.array([0], dtype=request.param)
    a[0] = 10
    return a


@skip_on_gpu
def test_print_scalar_with_string(input_arrays, capfd):
    """Tests if we can print a scalar value with a string."""

    if dpctl.SyclDevice().device_type == dpctl.device_type.cpu:
        pytest.xfail("Printing scalars on OpenCL CPU devices is unsupported.")

    @dpex.kernel
    def print_scalar_val(s):
        print("printing ...", s[0])

    a = input_arrays

    dpex.call_kernel(print_scalar_val, dpex.Range(1), a)
    captured = capfd.readouterr()
    assert "printing ... 10" in captured.out


@skip_on_gpu
def test_print_scalar(input_arrays, capfd):
    """Tests if we can print a scalar value."""

    if dpctl.SyclDevice().device_type == dpctl.device_type.cpu:
        pytest.xfail("Printing scalars on OpenCL CPU devices is unsupported.")

    @dpex.kernel
    def print_scalar_val(s):
        print(s[0])

    a = input_arrays

    dpex.call_kernel(print_scalar_val, dpex.Range(1), a)
    captured = capfd.readouterr()

    assert "10" in captured.out


@skip_on_gpu
def test_print_only_str(input_arrays):
    """Negative test to capture LoweringError as printing strings is
    unsupported.
    """

    @dpex.kernel
    def print_string(a):
        print("cannot print only a string inside a kernel")

    # This test will fail, we currently can not print only string.
    # The LLVM generated for printf() function with only string gets
    # replaced by a puts() which fails due to lack of addrspace in the
    # puts function signature right now, and would fail in general due
    # to lack of support for puts() in OpenCL.

    a = input_arrays

    with pytest.raises(TypingError) as ex_info:
        dpex.call_kernel(print_string, dpex.Range(1), a)

    assert "LoweringError" in ex_info.value.args[0]


@skip_on_gpu
def test_print_array(input_arrays):
    """Negative test to capture LoweringError as printing arrays
    is unsupported.
    """

    @dpex.kernel
    def print_string(a):
        print(a)

    a = input_arrays

    with pytest.raises(TypingError) as ex_info:
        dpex.call_kernel(print_string, dpex.Range(1), a)

    assert "LoweringError" in ex_info.value.args[0]
