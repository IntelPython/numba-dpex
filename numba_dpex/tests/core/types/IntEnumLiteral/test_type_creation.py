# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import pytest

from numba_dpex.core.exceptions import IllegalIntEnumLiteralValueError
from numba_dpex.core.types import IntEnumLiteral
from numba_dpex.kernel_api.flag_enum import FlagEnum


def test_intenumliteral_creation():
    """Tests the creation of an IntEnumLiteral type."""

    class DummyFlags(FlagEnum):
        DUMMY = 0

    try:
        IntEnumLiteral(DummyFlags)
    except:
        pytest.fail("Unexpected failure in IntEnumLiteral initialization")

    with pytest.raises(IllegalIntEnumLiteralValueError):

        class SomeKindOfUnknownEnum(IntEnum):
            UNKNOWN_FLAG = 1

        IntEnumLiteral(SomeKindOfUnknownEnum)
