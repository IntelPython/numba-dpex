# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import dpnp
import numpy as np

from numba_dpex.core.typing import dpnpdecl


def get_ufuncs():
    """obtain a list of supported ufuncs in the db"""

    from numba.np.ufunc_db import _lazy_init_db

    _lazy_init_db()
    from numba.np.ufunc_db import _ufunc_db

    _fill_ufunc_db_with_dpnp_ufuncs(_ufunc_db)

    return _ufunc_db.keys()


def _fill_ufunc_db_with_dpnp_ufuncs(ufunc_db):
    """Monkey patching dpnp for missing attributes."""
    # FIXME: add more docstring

    for ufuncop in dpnpdecl.supported_ufuncs:
        op = getattr(dpnp, ufuncop)
        npop = getattr(np, ufuncop)
        op.nin = npop.nin
        op.nout = npop.nout
        op.nargs = npop.nargs
        op.types = npop.types
        op.is_dpnp_ufunc = True
        ufunc_db.update({op: ufunc_db[npop]})
        for key in list(ufunc_db[op].keys()):
            if "FF->" in key or "DD->" in key or "F->" in key or "D->" in key:
                ufunc_db[op].pop(key)
