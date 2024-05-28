# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy

import dpnp
import numpy as np
from numba.core import types

from numba_dpex.core.typing import dpnpdecl
from numba_dpex.kernel_api_impl.spirv.math import mathimpl

# A global instance of dpnp ufuncs that are supported by numba-dpex
_dpnp_ufunc_db = None


def _lazy_init_dpnp_db():
    global _dpnp_ufunc_db

    if _dpnp_ufunc_db is None:
        _dpnp_ufunc_db = {}
        _fill_ufunc_db_with_dpnp_ufuncs(_dpnp_ufunc_db)


def get_ufuncs():
    """Returns the list of supported dpnp ufuncs in the _dpnp_ufunc_db"""

    _lazy_init_dpnp_db()

    return _dpnp_ufunc_db.keys()


def get_ufunc_info(ufunc_key):
    """get the lowering information for the ufunc with key ufunc_key.

    The lowering information is a dictionary that maps from a numpy
    loop string (as given by the ufunc types attribute) to a function
    that handles code generation for a scalar version of the ufunc
    (that is, generates the "per element" operation").

    raises a KeyError if the ufunc is not in the ufunc_db
    """
    _lazy_init_dpnp_db()
    return _dpnp_ufunc_db[ufunc_key]


def _fill_ufunc_db_with_dpnp_ufuncs(ufunc_db):
    """Populates the _dpnp_ufunc_db from Numba's NumPy ufunc_db"""

    from numba.np.ufunc_db import _lazy_init_db

    _lazy_init_db()

    # we need to import it after, because before init it is None and
    # variable is passed by value
    from numba.np.ufunc_db import _ufunc_db

    for ufuncop in dpnpdecl.supported_ufuncs:
        if ufuncop == "erf":
            op = getattr(dpnp, "erf")

            _unary_d_d = types.float64(types.float64)
            _unary_f_f = types.float32(types.float32)
            ufunc_db[op] = {
                "f->f": mathimpl.lower_ocl_impl[("erf", (_unary_f_f))],
                "d->d": mathimpl.lower_ocl_impl[("erf", (_unary_d_d))],
            }
        else:
            dpnpop = getattr(dpnp, ufuncop)
            npop = getattr(np, ufuncop)

            cp = copy.copy(_ufunc_db[npop])
            ufunc_db.update({dpnpop: cp})
            for key in list(ufunc_db[dpnpop].keys()):
                if (
                    "FF->" in key
                    or "DD->" in key
                    or "F->" in key
                    or "D->" in key
                ):
                    ufunc_db[dpnpop].pop(key)
