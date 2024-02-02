# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from numba.core.datamodel import default_manager

from numba_dpex.core.datamodel.models import dpex_data_model_manager
from numba_dpex.experimental import IntEnumLiteral
from numba_dpex.experimental.models import exp_dmm
from numba_dpex.kernel_api.flag_enum import FlagEnum


def test_data_model_registration():
    """Tests that the IntEnumLiteral type is only registered with the
    DpexExpKernelTargetContext target.
    """

    class DummyFlags(FlagEnum):
        DUMMY = 0

    dummy = IntEnumLiteral(DummyFlags)

    with pytest.raises(KeyError):
        default_manager.lookup(dummy)

    with pytest.raises(KeyError):
        dpex_data_model_manager.lookup(dummy)

    try:
        exp_dmm.lookup(dummy)
    except:
        pytest.fail(
            "IntEnumLiteral type lookup failed in experimental "
            "data model manager"
        )
