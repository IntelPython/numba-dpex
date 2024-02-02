# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Contains the ``box`` and ``unbox`` functions for numba_dpex types that are
passable as arguments to a kernel or dpjit decorated function.
"""

from .ranges import *
from .usm_ndarray import *
