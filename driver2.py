import dpnp
import numba
import numpy as np
from numba.core.typing import npydecl
from numba.np import npyimpl, ufunc_db

import numba_dpex as dpex

#  monkey patch dpnp's ufunc to have `nin
list_of_ufuncs = ["subtract"]


def fill_ufunc_db_with_dpnp_ufuncs():
    from numba.np.ufunc_db import _lazy_init_db

    _lazy_init_db()
    from numba.np.ufunc_db import _ufunc_db as ufunc_db

    for ufuncop in list_of_ufuncs:
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

    # from numba.np import npyfuncs
    # from numba.cpython import cmathimpl, mathimpl, numbers
    # from numba.np.numpy_support import numpy_version

    # dpnp.add.nin = np.add.nin
    # dpnp.add.nout = np.add.nout
    # dpnp.add.nargs = np.add.nargs
    # dpnp.add.types = np.add.types
    # dpnp.add.is_dpnp_ufunc = True
    # ufunc_db[dpnp.add] = {
    #     '??->?': numbers.int_or_impl,
    #     'bb->b': numbers.int_add_impl,
    #     'BB->B': numbers.int_add_impl,
    #     'hh->h': numbers.int_add_impl,
    #     'HH->H': numbers.int_add_impl,
    #     'ii->i': numbers.int_add_impl,
    #     'II->I': numbers.int_add_impl,
    #     'll->l': numbers.int_add_impl,
    #     'LL->L': numbers.int_add_impl,
    #     'qq->q': numbers.int_add_impl,
    #     'QQ->Q': numbers.int_add_impl,
    #     'ff->f': numbers.real_add_impl,
    #     'dd->d': numbers.real_add_impl,
    #     'FF->F': numbers.complex_add_impl,
    #     'DD->D': numbers.complex_add_impl,
    # }


def _register_dpnp_ufuncs():
    kernels = {}
    # NOTE: Assuming ufunc implementation for the CPUContext.
    for ufunc in ufunc_db.get_ufuncs():
        print("--->ufunc:", ufunc)
        # if ufunc == dpnp.add:
        #     breakpoint()
        kernels[ufunc] = npyimpl.register_ufunc_kernel(
            ufunc, npyimpl._ufunc_db_function(ufunc)
        )

    for _op_map in (
        npydecl.NumpyRulesUnaryArrayOperator._op_map,
        npydecl.NumpyRulesArrayOperator._op_map,
    ):
        for operator, ufunc_name in _op_map.items():
            if ufunc_name in list_of_ufuncs:
                ufunc = getattr(dpnp, ufunc_name)
                kernel = kernels[ufunc]
                if ufunc.nin == 1:
                    npyimpl.register_unary_operator_kernel(
                        operator, ufunc, kernel
                    )
                elif ufunc.nin == 2:
                    npyimpl.register_binary_operator_kernel(
                        operator, ufunc, kernel
                    )
                else:
                    raise RuntimeError(
                        "There shouldn't be any non-unary or binary operators"
                    )

    # for _op_map in (npydecl.NumpyRulesInplaceArrayOperator._op_map,):
    #     for operator, ufunc_name in _op_map.items():
    #         ufunc = getattr(np, ufunc_name)
    #         kernel = kernels[ufunc]
    #         if ufunc.nin == 1:
    #             npyimpl.register_unary_operator_kernel(
    #                 operator, ufunc, kernel, inplace=True
    #             )
    #         elif ufunc.nin == 2:
    #             npyimpl.register_binary_operator_kernel(
    #                 operator, ufunc, kernel, inplace=True
    #             )
    #         else:
    #             raise RuntimeError(
    #                 "There shouldn't be any non-unary or binary operators"
    #             )


fill_ufunc_db_with_dpnp_ufuncs()
_register_dpnp_ufuncs()


@dpex.dpjit
def foo1(a, b):
    return np.add(a, b)


@dpex.dpjit
def foo2(a, b):
    return dpnp.add(a, b)


a = dpnp.ones(10)
b = dpnp.ones(10)

c = np.ones(10)
d = np.ones(10)

# e = foo1(c, d)
f = foo2(a, b)
# print(c)
