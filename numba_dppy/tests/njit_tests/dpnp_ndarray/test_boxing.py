from numba_dppy.types import dpnp_ndarray_Type


def test_unbox_for_dpnp_ndarray_Type_should_be_registered():
    """Test that unboxing registered for dpnp_ndarray_Type"""
    from numba.core.pythonapi import _unboxers

    from numba_dppy.types.dpnp_boxing import unbox_array

    assert dpnp_ndarray_Type in _unboxers.functions
    assert unbox_array == _unboxers.lookup(dpnp_ndarray_Type)
