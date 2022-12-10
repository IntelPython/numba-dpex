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

from numba import types
from numba.core.datamodel import StructModel
from numba.core.datamodel.models import ArrayModel
from numba.extending import register_model

from numba_dpex.core.target import spirv_data_model_manager

from .dpnp_types import dpnp_ndarray_Type

# we reuse models.ArrayModel
# it should contain all properties from __sycl_usm_array_interface__
# __sycl_usm_array_interface__ and __array_interface__ are different
# _Memory.__sycl_usm_array_interface__ has data, shape, strides, typestr, version, syclobj


class dpnp_ndarray_Model(StructModel):
    """Model for DPNP array.

    It contains all items from models.ArrayModel and adding properties from
    __sycl_usm_array_interface__: syclobj
    """

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            # from models.ArrayModel
            ("meminfo", types.MemInfoPointer(fe_type.dtype)),
            ("parent", types.pyobject),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            ("data", types.CPointer(fe_type.dtype)),
            # from __sycl_usm_array_interface__
            ("syclobj", types.pyobject),
            # from models.ArrayModel
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
        ]
        super().__init__(dmm, fe_type, members)


# This tells Numba to use the default Numpy ndarray data layout for
# object of type dpnp.ndarray.
# register_model(dpnp_ndarray_Type)(ArrayModel)

register_model(dpnp_ndarray_Type)(dpnp_ndarray_Model)
spirv_data_model_manager.register(dpnp_ndarray_Type, ArrayModel)
