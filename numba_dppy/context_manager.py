from contextlib import contextmanager

import dpctl
from numba import njit
from numba._dispatcher import set_use_tls_target_stack
from numba.core.dispatcher import TargetConfig
from numba.core.retarget import BasicRetarget

TARGET = "SyclDevice"


def dppy_target(cpu_disp):
    kernel = njit(_target="SyclDevice")(cpu_disp.py_func)
    return kernel


class DPPYRetarget(BasicRetarget):
    @property
    def output_target(self):
        return TARGET

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target=TARGET)(cpu_disp.py_func)
        return kernel


retarget = DPPYRetarget()


@contextmanager
def offload_to_sycl_device(dpctl_device):
    with dpctl.device_context(dpctl_device):
        with TargetConfig.switch_target(retarget):
            yield
