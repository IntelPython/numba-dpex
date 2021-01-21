from numba.testing import SerialSuite
from numba.testing import load_testsuite
from os.path import dirname, join

import numba_dppy
import numba_dppy.config as dppy_config

# from numba_dppy.tests.dppy import *


def load_tests(loader, tests, pattern):

    suite = SerialSuite()

    if dppy_config.dppy_present:
        suite.addTests(load_testsuite(loader, dirname(__file__)))
    else:
        print("skipped DPPY tests")

    return suite
