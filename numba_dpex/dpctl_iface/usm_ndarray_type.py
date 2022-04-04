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

from dpctl.tensor import usm_ndarray
from numba.extending import register_model, typeof_impl
from numba.np import numpy_support

import numba_dpex.target as dpex_target
from numba_dpex.core.types import Array, ArrayModel
from numba_dpex.utils import address_space


class USMNdArrayType(Array):
    """
    USMNdArrayType(dtype, ndim, layout, usm_type,
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
        super(USMNdArrayType, self).__init__(
            dtype,
            ndim,
            layout,
            readonly=readonly,
            name=name,
            addrspace=addrspace,
        )

    def copy(self, *args, **kwargs):
        return super(USMNdArrayType, self).copy(*args, **kwargs)


# Tell Numba to use the numba_dpex.core.types.Array data layout for object of
# type USMNdArrayType.
register_model(USMNdArrayType)(ArrayModel)
dpex_target.spirv_data_model_manager.register(USMNdArrayType, ArrayModel)


@typeof_impl.register(usm_ndarray)
def typeof_usm_ndarray(val, c):
    """
    Creates the Numba type (USMNdArrayType) when a usm_ndarray is passed as an
    argument to a kernel.
    """
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
    layout = "C"
    readonly = False
    return USMNdArrayType(
        dtype,
        val.ndim,
        layout,
        val.usm_type,
        readonly=readonly,
        addrspace=address_space.GLOBAL,
    )
