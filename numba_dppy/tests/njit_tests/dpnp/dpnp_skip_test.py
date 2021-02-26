from numba_dppy.tests.skip_tests import skip_test
from numba_dppy.testing import ensure_dpnp

def dpnp_skip_test(device_type):
    skip = False
    if skip_test(device_type):
        skip = True

    if not skip:
        if not ensure_dpnp():
            skip = True

    return skip
