# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

def patch():
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
