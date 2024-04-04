# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a simple array intended to be used inside kernel work item.
Implementation is intended to be used in pure Python code when prototyping a
kernel function.
"""

import numpy as np


class PrivateArray:
    """An array that gets allocated on the private memory of a work-item.

    The class should be used to allocate small arrays on the private
    per-work-item memory for fast accesses inside a kernel. It is similar in
    intent to the :sycl_private_memory:`sycl::private_memory <>` class but is
    not a direct analogue.
    """

    def __init__(self, shape, dtype, fill_zeros=False) -> None:
        """Creates a new PrivateArray instance of the given shape and dtype."""

        if fill_zeros:
            self._data = np.zeros(shape=shape, dtype=dtype)
        else:
            self._data = np.empty(shape=shape, dtype=dtype)

    def __getitem__(self, idx_obj):
        """Returns the value stored at the position represented by idx_obj in
        the self._data ndarray.
        """

        return self._data[idx_obj]

    def __setitem__(self, idx_obj, val):
        """Assigns a new value to the position represented by idx_obj in
        the self._data ndarray.
        """

        self._data[idx_obj] = val
