# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a FlagEnum class to help distinguish IntEnum types that numba_dpex
intends to use as Integer literal types inside the compiler type inferring
infrastructure.
"""
from enum import IntEnum


class FlagEnum(IntEnum):
    """Helper class to distinguish IntEnum types that numba_dpex should consider
    as Numba Literal types.
    """

    @classmethod
    def basetype(cls) -> int:
        """Returns an dummy int object that helps numba_dpex infer the type of
        an instance of a FlagEnum class.

        Returns:
            int: Dummy int value
        """
        return int(0)
