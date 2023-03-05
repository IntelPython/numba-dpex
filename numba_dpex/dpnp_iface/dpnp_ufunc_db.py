# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import dpnp
import numpy as np
from numba.core import types

from numba_dpex.core.typing import dpnpdecl

from ..ocl import mathimpl


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
        if ufuncop == "erf":
            op = getattr(dpnp, "erf")
            op.nin = 1
            op.nout = 1
            op.nargs = 2
            op.types = ["f->f", "d->d"]
            op.is_dpnp_ufunc = True

            _unary_d_d = types.float64(types.float64)
            _unary_f_f = types.float32(types.float32)
            ufunc_db[op] = {
                "f->f": mathimpl.lower_ocl_impl[("erf", (_unary_f_f))],
                "d->d": mathimpl.lower_ocl_impl[("erf", (_unary_d_d))],
            }
        else:
            op = getattr(dpnp, ufuncop)
            npop = getattr(np, ufuncop)
            op.nin = npop.nin
            op.nout = npop.nout
            op.nargs = npop.nargs
            op.types = npop.types
            op.is_dpnp_ufunc = True
            ufunc_db.update({op: ufunc_db[npop]})
            for key in list(ufunc_db[op].keys()):
                if (
                    "FF->" in key
                    or "DD->" in key
                    or "F->" in key
                    or "D->" in key
                ):
                    ufunc_db[op].pop(key)
