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
    """
    USM_NdArrayType(dtype, ndim, layout, usm_type,
                    readonly=False, name=None,
                    aligned=True, addrspace=None)
    creates Numba type to represent ``dpctl.tensor.usm_ndarray``.
    """

    def __init__(
        self,
        dtype,
        ndim,
        layout,
        usm_type,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
    ):
        self.usm_type = usm_type
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
        return super(USM_NdArrayType, self).copy(*args, **kwargs)


# This tells Numba to use the DPPYArray data layout for object of type USM_NdArrayType.

register_model(USM_NdArrayType)(DPPYArrayModel)
dppy_target.spirv_data_model_manager.register(USM_NdArrayType, DPPYArrayModel)


@typeof_impl.register(usm_ndarray)
def typeof_usm_ndarray(val, c):
    """
    This function creates the Numba type (USM_NdArrayType) when a usm_ndarray is passed.
    """
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
    layout = "C"
    readonly = False
    return USM_NdArrayType(dtype, val.ndim, layout, val.usm_type, readonly=readonly)
