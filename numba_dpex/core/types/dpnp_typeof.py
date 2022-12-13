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

from numba.extending import typeof_impl
from numba.np import numpy_support

from .dpnp_types import dpnp_ndarray_Type, ndarray


# This tells Numba how to create a UsmSharedArrayType when a usmarray is passed
# into a njit function.
@typeof_impl.register(ndarray)
def typeof_dpnp_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))

    try:
        layout = numpy_support.map_layout(val)
    except AttributeError:
        try:
            # passing nested object as dpnp.ndarray does not support flags yet
            layout = numpy_support.map_layout(val._array_obj)
        except TypeError:
            layout = "C"

    try:
        readonly = not val.flags.writeable
    except AttributeError:
        # dpnp.ndarray does not support flags
        readonly = False

    sycl_queue = val.device
    return dpnp_ndarray_Type(
        dtype, val.ndim, layout, readonly=readonly, sycl_queue=sycl_queue
    )
