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

from numba.extending import typeof_impl, register_model
from numba_dppy.dppy_array_type import DPPYArray, DPPYArrayModel
import numba_dppy.target as dppy_target
from dpctl.tensor import usm_ndarray
from numba.np import numpy_support


class USM_NdArrayType(DPPYArray):
    def __init__(
        self,
        dtype,
        ndim,
        layout,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
    ):
        # This name defines how this type will be shown in Numba's type dumps.
        name = "USM:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
        super(USM_NdArrayType, self).__init__(
            dtype,
            ndim,
            layout,
            py_type=usm_ndarray,
            readonly=readonly,
            name=name,
            addrspace=addrspace,
        )

    def copy(self, *args, **kwargs):
        retty = super(USM_NdArrayType, self).copy(*args, **kwargs)
        if isinstance(retty, types.Array):
            return USM_NdArrayType(
                dtype=retty.dtype, ndim=retty.ndim, layout=retty.layout
            )
        else:
            return retty


# This tells Numba to use the default Numpy ndarray data layout for
# object of type UsmArray.
register_model(USM_NdArrayType)(DPPYArrayModel)
dppy_target.spirv_data_model_manager.register(USM_NdArrayType, DPPYArrayModel)

# This tells Numba how to create a UsmSharedArrayType when a usmarray is passed
# into a njit function.
@typeof_impl.register(usm_ndarray)
def typeof_ta_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
    layout = "C"
    readonly = False
    return USM_NdArrayType(dtype, val.ndim, layout, readonly=readonly)
