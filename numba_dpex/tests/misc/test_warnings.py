import logging

import dpnp

import numba_dpex as dpex
import numba_dpex.config as config


@dpex.kernel(enable_cache=False)
def foo(a):
    a[dpex.get_global_id(0)] = 0


def test_opt_warning(caplog):
    bkp = config.DPEX_OPT
    config.DPEX_OPT = 3

    with caplog.at_level(logging.WARNING):
        foo[dpex.Range(10)](dpnp.arange(10))

    config.DPEX_OPT = bkp

    assert "NUMBA_DPEX_OPT" in caplog.text


def test_inline_warning(caplog):
    bkp = config.INLINE_THRESHOLD
    config.INLINE_THRESHOLD = 2

    with caplog.at_level(logging.WARNING):
        foo[dpex.Range(10)](dpnp.arange(10))

    config.INLINE_THRESHOLD = bkp

    assert "INLINE_THRESHOLD" in caplog.text


def test_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        foo[dpex.Range(10)](dpnp.arange(10))

    assert caplog.text == ""
