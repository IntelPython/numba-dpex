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

"""Support for interoperability."""

import dpctl.tensor as dpt


def asarray(container):
    """Convert container supported by interoperability to numba-dppy container.
    Currently used dpctl.tensor.asarray().
    """
    try:
        return dpt.asarray(container)
    except:
        pass

    # Workaround for dpnp_array if dpctl asarray() does not support it.
    try:
        from dpnp.dpnp_array import dpnp_array

        if isinstance(container, dpnp_array) and hasattr(
            container, "_array_obj"
        ):
            import warnings

            warnings.warn("asarray() uses internals from dpnp.")
            return container._array_obj
    except:
        pass

    raise NotImplementedError(
        "dpctl asarray() does not support " + type(container)
    )
