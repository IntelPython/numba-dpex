# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from llvmlite import ir
from numba.core import types
from numba.core.extending import register_jitable
from numba.core.imputils import lower_getattr
from numba.cpython import listobj

ll_void_p = ir.IntType(8).as_pointer()


def get_dpnp_fptr(fn_name, type_names):
    from . import dpnp_fptr_interface as dpnp_iface

    f_ptr = dpnp_iface.get_dpnp_fn_ptr(fn_name, type_names)
    return f_ptr


@register_jitable
def _check_finite_matrix(a):
    for v in np.nditer(a):
        if not np.isfinite(v.item()):
            raise np.linalg.LinAlgError("Array must not contain infs or NaNs.")


@register_jitable
def _dummy_liveness_func(a):
    """pass a list of variables to be preserved through dead code elimination"""
    return a[0]


def dpnp_func(fn_name, type_names, sig):
    f_ptr = get_dpnp_fptr(fn_name, type_names)

    def get_pointer(obj):
        return f_ptr

    return types.ExternalFunctionPointer(sig, get_pointer=get_pointer)


"""
This function retrieves the pointer to the structure where the shape
of an ndarray is stored. We cast it to void * to make it easier to
pass around.
"""


@lower_getattr(types.Array, "shapeptr")
def array_shapeptr(context, builder, typ, value):
    shape_ptr = builder.gep(
        value.operands[0],
        [
            context.get_constant(types.int32, 0),
            context.get_constant(types.int32, 5),
        ],
    )

    return builder.bitcast(shape_ptr, ll_void_p)


@lower_getattr(types.List, "size")
def list_size(context, builder, typ, value):
    inst = listobj.ListInstance(context, builder, typ, value)
    return inst.size


@lower_getattr(types.List, "itemsize")
def list_itemsize(context, builder, typ, value):
    llty = context.get_data_type(typ.dtype)
    return context.get_constant(types.uintp, context.get_abi_sizeof(llty))


@lower_getattr(types.List, "ctypes")
def list_ctypes(context, builder, typ, value):
    inst = listobj.ListInstance(context, builder, typ, value)
    return builder.bitcast(inst.data, ll_void_p)
