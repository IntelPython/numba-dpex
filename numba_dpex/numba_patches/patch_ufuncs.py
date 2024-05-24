# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging

import dpnp
import numpy as np

from numba_dpex.core.typing import dpnpdecl


def patch():
    patch_is_ufunc()
    patch_ufuncs()


def patch_is_ufunc():
    """Patches the numba.np.ufunc.array_exprs._is_ufunc function to make it
    possible to support dpnp universal functions (ufuncs).

    The extra condition is the check for the "is_dpnp_ufunc" attribute to
    identify a non-NumPy ufunc.
    """
    import numpy
    from numba.np.ufunc.dufunc import DUFunc

    def _is_ufunc(func):
        return isinstance(func, (numpy.ufunc, DUFunc)) or hasattr(
            func, "is_dpnp_ufunc"
        )

    from numba.np.ufunc import array_exprs

    array_exprs._is_ufunc = _is_ufunc


def patch_ufuncs():
    """Patches dpnp user functions to make them compatible with numpy, so we
    can reuse numba's implementation.

    It adds "nin", "nout", "nargs" and "is_dpnp_ufunc" attributes to ufuncs.
    """
    failed_dpnpop_types_lst = []

    op = getattr(dpnp, "erf")
    op.nin = 1
    op.nout = 1
    op.nargs = 2
    op.types = ["f->f", "d->d"]
    op.is_dpnp_ufunc = True

    for ufuncop in dpnpdecl.supported_ufuncs:
        if ufuncop == "erf":
            continue

        dpnpop = getattr(dpnp, ufuncop)
        npop = getattr(np, ufuncop)

        if not hasattr(dpnpop, "nin"):
            dpnpop.nin = npop.nin
        if not hasattr(dpnpop, "nout"):
            dpnpop.nout = npop.nout
        if not hasattr(dpnpop, "nargs"):
            dpnpop.nargs = dpnpop.nin + dpnpop.nout

        # Check if the dpnp operation has a `types` attribute and if an
        # AttributeError gets raised then "monkey patch" the attribute from
        # numpy. If the attribute lookup raised a ValueError, it indicates
        # that dpnp could not be resolve the supported types for the
        # operation. Dpnp will fail to resolve the `types` if no SYCL
        # devices are available on the system. For such a scenario, we log
        # dpnp operations for which the ValueError happened and print them
        # as a user-level warning. It is done this way so that the failure
        # to load the dpnpdecl registry due to the ValueError does not
        # impede a user from importing numba-dpex.
        try:
            dpnpop.types
        except ValueError:
            failed_dpnpop_types_lst.append(ufuncop)
        except AttributeError:
            dpnpop.types = npop.types

        dpnpop.is_dpnp_ufunc = True

    if len(failed_dpnpop_types_lst) > 0:
        try:
            getattr(dpnp, failed_dpnpop_types_lst[0]).types
        except ValueError:
            ops = " ".join(failed_dpnpop_types_lst)
            logging.exception(
                "The types attribute for the following dpnp ops could not be "
                f"determined: {ops}"
            )
