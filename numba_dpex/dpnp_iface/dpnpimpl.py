# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numba.np import npyimpl

from numba_dpex.core.typing.dpnpdecl import _unsupported
from numba_dpex.dpnp_iface import dpnp_ufunc_db


def _register_dpnp_ufuncs():
    kernels = {}
    # NOTE: Assuming ufunc implementation for the CPUContext.
    for ufunc in dpnp_ufunc_db.get_ufuncs():
        kernels[ufunc] = npyimpl.register_ufunc_kernel(
            ufunc, npyimpl._ufunc_db_function(ufunc)
        )

    for _op_map in (
        npyimpl.npydecl.NumpyRulesUnaryArrayOperator._op_map,
        npyimpl.npydecl.NumpyRulesArrayOperator._op_map,
    ):
        for operator, ufunc_name in _op_map.items():
            if ufunc_name in _unsupported:
                continue
            ufunc = getattr(dpnp, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                npyimpl.register_unary_operator_kernel(operator, ufunc, kernel)
            elif ufunc.nin == 2:
                npyimpl.register_binary_operator_kernel(operator, ufunc, kernel)
            else:
                raise RuntimeError(
                    "There shouldn't be any non-unary or binary operators"
                )

    for _op_map in (npyimpl.npydecl.NumpyRulesInplaceArrayOperator._op_map,):
        for operator, ufunc_name in _op_map.items():
            if ufunc_name in _unsupported:
                continue
            ufunc = getattr(dpnp, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                npyimpl.register_unary_operator_kernel(
                    operator, ufunc, kernel, inplace=True
                )
            elif ufunc.nin == 2:
                npyimpl.register_binary_operator_kernel(
                    operator, ufunc, kernel, inplace=True
                )
            else:
                raise RuntimeError(
                    "There shouldn't be any non-unary or binary operators"
                )


_register_dpnp_ufuncs()
