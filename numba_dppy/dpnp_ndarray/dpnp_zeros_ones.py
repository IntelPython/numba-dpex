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
import dpnp
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import (
    intrinsic,
    lower_builtin,
    overload_classmethod,
    type_callable,
)
from numba_dppy.dpctl_iface.dpctl_capi_fn_builder import DpctlCAPIFnBuilder

from .types import dpnp_ndarray_Type
from .dpnp_empty import parse_usm_type, _parse_empty_args, _empty_nd_impl


@type_callable(dpnp.zeros)
@type_callable(dpnp.ones)
def type_dpnp_empty(context):
    def typer(shape, dtype=None, usm_type=None, sycl_queue=None):
        from numba.core.typing.npydecl import parse_dtype, parse_shape

        if dtype is None:
            nb_dtype = types.double
        else:
            nb_dtype = parse_dtype(dtype)

        ndim = parse_shape(shape)

        if usm_type is None:
            usm_type = "device"
        else:
            usm_type = parse_usm_type(usm_type)

        if nb_dtype is not None and ndim is not None and usm_type is not None:
            return dpnp_ndarray_Type(
                dtype=nb_dtype, ndim=ndim, layout="C", usm_type=usm_type
            )

    return typer


@lower_builtin(dpnp.zeros, types.Any, types.Any, types.Any, types.Any)
def impl_dpnp_zeros(context, builder, sig, args):
    """
    Inputs: shape, dtype, usm_type, queue
    """
    from numba.core.imputils import impl_ret_new_ref

    empty_args = _parse_empty_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, *empty_args)
    _zero_fill_nd(context, builder, ary)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())


def _zero_fill_nd(context, builder, ary):
    _memset(context, builder, ary.data, builder.mul(ary.itemsize, ary.nitems), 0, ary.queue)


def _memset(context, builder, ptr, size, value, queue):
    """
    Fill *size* bytes starting from *ptr* with *value*.
    """
    from numba.core.cgutils import int8_t, voidptr_t

    memset_fn = DpctlCAPIFnBuilder.get_dpctl_queue_memset(
                    builder=builder, context=context)
    event_wait_fn = DpctlCAPIFnBuilder.get_dpctl_queue_wait(
                    builder=builder, context=context)

    ptr = builder.bitcast(ptr, voidptr_t)
    if isinstance(value, int):
        value = int8_t(value)

    event_ref = builder.call(memset_fn, queue, ptr, size, value)
    builder.call(event_wait_fn, [event_ref])


# TODO: implement _ones_fill_nd
