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

from dpnp import ndarray
from numba.core import types

from .array_type import Array


class dpnp_ndarray_Type(Array):
    """Numba type for dpnp.ndarray."""

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
        name = "dpnp.ndarray(%s, %sd, %s)" % (dtype, ndim, layout)

        super().__init__(
            dtype,
            ndim,
            layout,
            # py_type=ndarray,
            readonly=readonly,
            name=name,
            addrspace=addrspace,
        )

    @property
    def as_array(self):
        return self

    @property
    def box_type(self):
        return ndarray
