from numba.core import types
from numba.core.datamodel import default_manager, models

from numba_dpex.core.types import dpnp_ndarray_Type
from numba_dpex.core.types.dpnp_models import dpnp_ndarray_Model


def test_model_for_dpnp_ndarray_Type():
    """Test that model is registered for dpnp_ndarray_Type instances.

    The model for dpnp_ndarray_Type is dpnp_ndarray_Model.
    It contains property "syclobj".
    This property is PyObject.
    """

    model = default_manager.lookup(dpnp_ndarray_Type(types.float64, 1, "C"))
    assert isinstance(model, dpnp_ndarray_Model)

    assert "syclobj" in model._fields
    assert model.get_member_fe_type("syclobj") == types.pyobject


def test_dpnp_ndarray_Model():
    """Test for dpnp_ndarray_Model.

    It is sumclass of models.StructModel and not a subclass of models.ArrayModel.
    """

    assert issubclass(dpnp_ndarray_Model, models.StructModel)
    assert not issubclass(dpnp_ndarray_Model, models.ArrayModel)
