import warnings

import dpnp
import pytest

import numba_dpex as dpex
import numba_dpex.config as config


@dpex.kernel(enable_cache=False)
def foo(a):
    a[dpex.get_global_id(0)] = 0


def test_opt_warning(caplog):
    bkp = config.DPEX_OPT
    config.DPEX_OPT = 3

    with pytest.warns(UserWarning):
        foo[dpex.Range(10)](dpnp.arange(10))

    config.DPEX_OPT = bkp


def test_inline_warning(caplog):
    bkp = config.INLINE_THRESHOLD
    config.INLINE_THRESHOLD = 3

    with pytest.warns(UserWarning):
        foo[dpex.Range(10)](dpnp.arange(10))

    config.INLINE_THRESHOLD = bkp


def test_no_warning(caplog):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        foo[dpex.Range(10)](dpnp.arange(10))
