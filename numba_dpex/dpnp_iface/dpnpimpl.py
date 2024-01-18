# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy

import dpnp
from numba.core.imputils import Registry
from numba.np import npyimpl

from numba_dpex.core.typing import dpnpdecl
from numba_dpex.dpnp_iface import dpnp_ufunc_db

registry = Registry("dpnpimpl")


def _register_dpnp_ufuncs():
    """Adds dpnp ufuncs to the dpnpimpl.registry.

    The npyimpl.registry is searched for all registered ufuncs and we copy the
    implementations and register them in a dpnp-specific registry defined in the
    current module. The numpy ufuncs are deep copied so as to not mutate the
    original functions by changes we introduce in the DpexKernelTarget.

    Raises:
        RuntimeError: If the signature of the ufunc takes more than two input
        args.
    """
    kernels = {}

    for ufunc in dpnp_ufunc_db.get_ufuncs():
        kernels[ufunc] = npyimpl.register_ufunc_kernel(
            ufunc,
            copy.copy(npyimpl._ufunc_db_function(ufunc)),
            registry.lower,
        )

    for _op_map in (
        dpnpdecl.DpnpRulesUnaryArrayOperator._op_map,
        dpnpdecl.DpnpRulesArrayOperator._op_map,
    ):
        for operator, ufunc_name in _op_map.items():
            if ufunc_name in dpnpdecl._unsupported:
                continue
            ufunc = getattr(dpnp, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                npyimpl.register_unary_operator_kernel(
                    operator, ufunc, kernel, registry.lower
                )
            elif ufunc.nin == 2:
                npyimpl.register_binary_operator_kernel(
                    operator, ufunc, kernel, registry.lower
                )
            else:
                raise RuntimeError(
                    "There shouldn't be any non-unary or binary operators"
                )

    for _op_map in (dpnpdecl.DpnpRulesInplaceArrayOperator._op_map,):
        for operator, ufunc_name in _op_map.items():
            if ufunc_name in dpnpdecl._unsupported:
                continue
            ufunc = getattr(dpnp, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                npyimpl.register_unary_operator_kernel(
                    operator, ufunc, kernel, registry.lower, inplace=True
                )
            elif ufunc.nin == 2:
                npyimpl.register_binary_operator_kernel(
                    operator, ufunc, kernel, registry.lower, inplace=True
                )
            else:
                raise RuntimeError(
                    "There shouldn't be any non-unary or binary operators"
                )


# Initialize the registry that stores the dpnp ufuncs
_register_dpnp_ufuncs()
