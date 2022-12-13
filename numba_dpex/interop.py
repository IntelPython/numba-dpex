# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Support for interoperability."""

import dpctl.tensor as dpt


def asarray(container):
    """A wrapper over dpctl.tensor asarray function to convert any array
    that supports the ``__sycl_usm_array_interface__`` protocol to a
    ``dpctl.tensor.usm_ndarray``.
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
