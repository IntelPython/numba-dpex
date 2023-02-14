# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import cgutils
from numba.core.errors import NumbaNotImplementedError
from numba.core.pythonapi import NativeValue, PythonAPI, box, unbox
from numba.np import numpy_support

from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.runtime import context as dpexrt

from .usm_ndarray_type import USMNdArray


class DpnpNdArray(USMNdArray):
    """
    The Numba type to represent an dpnp.ndarray. The type has the same
    structure as USMNdArray used to represent dpctl.tensor.usm_ndarray.
    """

    @property
    def is_internal(self):
        return True

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            if not all(isinstance(inp, DpnpNdArray) for inp in inputs):
                return NotImplemented
            return DpnpNdArray
        else:
            return


# --------------- Boxing/Unboxing logic for dpnp.ndarray ----------------------#


@unbox(DpnpNdArray)
def unbox_dpnp_nd_array(typ, obj, c):
    """Converts a dpnp.ndarray object to a Numba internal array structure.

    Args:
        typ : The Numba type of the PyObject
        obj : The actual PyObject to be unboxed
        c :

    Returns:
        _type_: _description_
    """
    # Reusing the numba.core.base.BaseContext's make_array function to get a
    # struct allocated. The same struct is used for numpy.ndarray
    # and dpnp.ndarray. It is possible to do so, as the extra information
    # specific to dpnp.ndarray such as sycl_queue is inferred statically and
    # stored as part of the DpnpNdArray type.

    # --------------- Original Numba comment from @ubox(types.Array)
    #
    # This is necessary because unbox_buffer() does not work on some
    # dtypes, e.g. datetime64 and timedelta64.
    # TODO check matching dtype.
    #      currently, mismatching dtype will still work and causes
    #      potential memory corruption
    #
    # --------------- End of Numba comment from @ubox(types.Array)
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()

    ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
    # FIXME : We need to check if Numba_RT as well as DPEX RT are enabled.
    if c.context.enable_nrt:
        dpexrtCtx = dpexrt.DpexRTContext(c.context)
        errcode = dpexrtCtx.arraystruct_from_python(c.pyapi, obj, ptr)
    else:
        raise UnreachableError

    # TODO: here we have minimal typechecking by the itemsize.
    #       need to do better
    try:
        expected_itemsize = numpy_support.as_dtype(typ.dtype).itemsize
    except NumbaNotImplementedError:
        # Don't check types that can't be `as_dtype()`-ed
        itemsize_mismatch = cgutils.false_bit
    else:
        expected_itemsize = nativeary.itemsize.type(expected_itemsize)
        itemsize_mismatch = c.builder.icmp_unsigned(
            "!=",
            nativeary.itemsize,
            expected_itemsize,
        )

    failed = c.builder.or_(
        cgutils.is_not_null(c.builder, errcode),
        itemsize_mismatch,
    )
    # Handle error
    with c.builder.if_then(failed, likely=False):
        c.pyapi.err_set_string(
            "PyExc_TypeError",
            "can't unbox array from PyObject into "
            "native value.  The object maybe of a "
            "different type",
        )
    return NativeValue(c.builder.load(aryptr), is_error=failed)


@box(DpnpNdArray)
def box_array(typ, val, c):
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        dpexrtCtx = dpexrt.DpexRTContext(c.context)
        newary = dpexrtCtx.usm_ndarray_to_python_acqref(
            c.pyapi, typ, val, dtypeptr
        )

        if not newary:
            c.pyapi.err_set_string(
                "PyExc_TypeError",
                "could not box native array into a dpnp.ndarray PyObject.",
            )

        # Steals NRT ref
        # Refer:
        #   numba.core.base.nrt -> numba.core.runtime.context -> decref
        #   The `NRT_decref` function is generated directly as LLVM IR inside
        #   numba.core.runtime.nrtdynmod.py
        c.context.nrt.decref(c.builder, typ, val)

        return newary
    else:
        raise UnreachableError
