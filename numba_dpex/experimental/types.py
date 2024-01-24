# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental types that will eventually move to numba_dpex.core.types
"""

from numba.core import types


class KernelDispatcherType(types.Dispatcher):
    """The type of KernelDispatcher dispatchers"""

    def cast_python_value(self, args):
        raise NotImplementedError
