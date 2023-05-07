import dpctl
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit
from numba_dpex.tests.core.types.DpctlSyclQueue._helper import are_queues_equal
from numba_dpex.tests.njit_tests.dpnp._helper import compile_function

dpnp_functions_params = [
    ("empty", [10]),
    ("zeros", [10]),
    ("ones", [10]),
    ("full", [10, 3]),
    ("empty_like", ["x"]),
    ("zeros_like", ["x"]),
    ("ones_like", ["x"]),
    ("full_like", ["x", 3]),
]

queue_args = ["", "sycl_queue=None", "sycl_queue=queue"]

func_template_basic = """
def {0:s}(x=None, queue=None):
    a = dpnp.{1:s}({2:s})
    return a
"""


@pytest.mark.parametrize("dpnp_function_param", dpnp_functions_params)
@pytest.mark.parametrize("queue_arg", queue_args)
def test_dpnp_functions_basic(dpnp_function_param, queue_arg):
    """Basic queue handling in a basic dpnp function scenario"""
    function_name = "func"
    op_name = dpnp_function_param[0]
    dpnpf_arg = ", ".join(str(v) for v in dpnp_function_param[1])
    q_arg = (", " + queue_arg) if queue_arg != "" else ""
    func_expr = func_template_basic.format(
        function_name, op_name, dpnpf_arg + q_arg
    )

    func = compile_function(func_expr, function_name, globals())
    f = dpjit(func)

    sycl_queue = dpctl.SyclQueue()
    x = dpnp.random.random(10) if dpnp_function_param[1][0] == "x" else None
    queue = sycl_queue if queue_arg == "sycl_queue=queue" else None

    a = f(x, queue)

    assert not are_queues_equal(sycl_queue, a.sycl_queue)

    device = dpctl.SyclDevice()
    sycl_queue = dpctl._sycl_queue_manager.get_device_cached_queue(device)
    assert are_queues_equal(sycl_queue, a.sycl_queue)
