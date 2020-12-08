from __future__ import print_function, absolute_import, division

import contextlib
import sys

from numba.core import config
import unittest
from numba.tests.support import (
    captured_stdout,
    SerialMixin,
    redirect_c_stdout,
)

class DPPYTestCase(SerialMixin, unittest.TestCase):
    def setUp(self):
        #init()
	#TODO
        pass
    def tearDown(self):
        #reset()
	#TODO
        pass

class DPPYTextCapture(object):
    def __init__(self, stream):
        self._stream = stream

    def getvalue(self):
        return self._stream.read()

class PythonTextCapture(object):
    def __init__(self, stream):
        self._stream = stream

    def getvalue(self):
        return self._stream.getvalue()

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
