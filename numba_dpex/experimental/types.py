# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import types


class KernelDispatcherType(types.Dispatcher):
    """The type of KernelDispatcher dispatchers"""
