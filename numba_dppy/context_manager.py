from contextlib import contextmanager

import dpctl
from numba import njit
from numba._dispatcher import set_use_tls_target_stack
from numba.core.dispatcher import TargetConfig
from numba.core.retarget import BasicRetarget

TARGET = "SyclDevice"


class DPPYRetarget(BasicRetarget):
    def __init__(self, filter_str):
        self.filter_str = filter_str
        super(DPPYRetarget, self).__init__()

    @property
    def output_target(self):
        return TARGET

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target=TARGET)(cpu_disp.py_func)
        return kernel


first_level_cache = dict()


@contextmanager
def offload_to_sycl_device(dpctl_device):
    with dpctl.device_context(dpctl_device):
        retarget = first_level_cache.get(dpctl_device.filter_string, None)

        if retarget is None:
            retarget = DPPYRetarget(dpctl_device.filter_string)
            first_level_cache[dpctl_device.filter_string] = retarget
        with TargetConfig.switch_target(retarget):
            yield
