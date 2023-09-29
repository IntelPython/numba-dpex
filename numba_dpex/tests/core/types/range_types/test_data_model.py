# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from numba.core.datamodel import default_manager

from numba_dpex.core.datamodel.models import (
    NdRangeModel,
    RangeModel,
    dpex_data_model_manager,
)
from numba_dpex.core.types.range_types import NdRangeType, RangeType

rfields = ["ndim", "dim0", "dim1", "dim2"]
ndrfields = ["ndim", "gdim0", "gdim1", "gdim2", "ldim0", "ldim1", "ldim2"]


def test_datamodel_registration():
    """Test the datamodel for RangeType and NdRangeType are found in numba's
    default datamodel manager but not in numba_dpex's kernel data model manager.
    """
    range_ty = RangeType(ndim=1)
    ndrange_ty = NdRangeType(ndim=1)

    with pytest.raises(KeyError):
        dpex_data_model_manager.lookup(range_ty)
        dpex_data_model_manager.lookup(ndrange_ty)

    default_range_model = default_manager.lookup(range_ty)
    default_ndrange_model = default_manager.lookup(ndrange_ty)

    assert isinstance(default_range_model, RangeModel)
    assert isinstance(default_ndrange_model, NdRangeModel)


@pytest.mark.parametrize("field", rfields)
def test_range_model_fields(field):
    """Tests that the expected fields are found in the data model for
    RangeType
    """
    range_ty = RangeType(ndim=1)
    dm = default_manager.lookup(range_ty)
    try:
        dm.get_field_position(field)
    except:
        pytest.fail(f"Expected field {field} not present in RangeModel")


@pytest.mark.parametrize("field", ndrfields)
def test_ndrange_model_fields(field):
    """Tests that the expected fields are found in the data model for
    NdRangeType
    """
    ndrange_ty = NdRangeType(ndim=1)
    dm = default_manager.lookup(ndrange_ty)
    try:
        dm.get_field_position(field)
    except:
        pytest.fail(f"Expected field {field} not present in NdRangeModel")
