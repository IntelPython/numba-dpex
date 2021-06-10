from contextlib import contextmanager

import dpctl
from numba import njit
from numba._dispatcher import set_use_tls_target_stack
from numba.core.dispatcher import TargetConfig



def dppy_target(cpu_disp):
    kernel = njit(_target="SyclDevice")(cpu_disp.py_func)
    return kernel

@contextmanager
def offload_to_sycl_device(dpctl_device):
    # __enter__

    with dpctl.device_context(dpctl_device):
        tc = TargetConfig()
        tc.push(dppy_target)
        set_use_tls_target_stack(True)
        yield
        # __exit__
        tc.pop()
        set_use_tls_target_stack(False)
