# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a simple array intended to be used inside kernel work item.
Implementation is intended to be used in pure Python code when prototyping a
kernel function.
"""

from numpy import ndarray


class PrivateArray:
    """
    The ``PrivateArray`` class is an simple version of array intended to be used
    inside kernel work item.
    """

    def __init__(self, shape, dtype) -> None:
        """Creates a new PrivateArray instance of the given shape and dtype."""

        self._data = ndarray(shape=shape, dtype=dtype)

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
