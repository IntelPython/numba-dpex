import contextlib
import sys

from numba.core import config
import unittest
from numba.tests.support import (
    captured_stdout,
    redirect_c_stdout,
)


@contextlib.contextmanager
def captured_dppy_stdout():
    """
    Return a minimal stream-like object capturing the text output of dppy
    """
    # Prevent accidentally capturing previously output text
    sys.stdout.flush()

    import numba_dppy, numba_dppy as dppy

    with redirect_c_stdout() as stream:
        yield DPPYTextCapture(stream)


def _id(obj):
    return obj


def expectedFailureIf(condition):
    """
    Expected failure for a test if the condition is true.
    """
    if condition:
        return unittest.expectedFailure
    return _id


def ensure_dpnp():
    try:
        from numba_dppy.dpnp_glue import dpnp_fptr_interface as dpnp_glue

        return True
    except:
        return False

def set_dpnp_debug(opt):
    import numba_dppy.dpnp_glue as dpnp_lowering

    dpnp_lowering.DEBUG = opt
