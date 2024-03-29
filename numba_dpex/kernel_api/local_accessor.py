# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a Python analogue to SYCL's local_accessor class. The class is
intended to be used in pure Python code when prototyping a kernel function
and to be passed to an actual kernel function for local memory allocation.
"""
import numpy


class LocalAccessor:
    """Analogue to the :sycl_local_accessor:`sycl::local_accessor <>` class.

    The class acts as a proxy to allocating device local memory and
    accessing that memory from within a :func:`numba_dpex.kernel` decorated
    function.
    """

    def _verify_positive_integral_list(self, ls):
        """Checks if all members of a list are positive integers."""

        ret = False
        try:
            ret = all(int(val) > 0 for val in ls)
        except ValueError:
            pass

        return ret

    def __init__(self, shape, dtype) -> None:
        """Creates a new LocalAccessor instance of the given shape and dtype."""

        if not isinstance(shape, (list, tuple)):
            if hasattr(shape, "tolist"):
                fn = getattr(shape, "tolist")
                if callable(fn):
                    self._shape = tuple(shape.tolist())
            else:
                try:
                    self._shape = (shape,)
                except Exception as e:
                    raise TypeError(
                        "Argument shape must a non-negative integer, "
                        "or a list/tuple of such integers."
                    ) from e
        else:
            self._shape = tuple(shape)

        # Make sure shape is made up a supported types
        if not self._verify_positive_integral_list(self._shape):
            raise TypeError(
                "Argument shape must a non-negative integer, "
                "or a list/tuple of such integers."
            )

        # Make sure shape has a rank between (1..3)
        if len(self._shape) < 1 or len(self._shape) > 3:
            raise TypeError("LocalAccessor can only have up to 3 dimensions.")

        self._dtype = dtype

        if self._dtype not in [
            numpy.float32,
            numpy.float64,
            numpy.int32,
            numpy.int64,
            numpy.int16,
            numpy.int8,
            numpy.uint32,
            numpy.uint64,
            numpy.uint16,
            numpy.uint8,
        ]:
            raise TypeError(
                f"Argument dtype {dtype} is not supported. numpy.float32, "
                "numpy.float64, numpy.[u]int8, numpy.[u]int16, numpy.[u]int32, "
                "numpy.[u]int64 are the currently supported dtypes."
            )

        self._data = numpy.empty(self._shape, dtype=self._dtype)

    def __getitem__(self, idx_obj):
        """Returns the value stored at the position represented by idx_obj in
        the self._data ndarray.
        """

        raise NotImplementedError(
            "The data of a LocalAccessor object can only be accessed "
            "inside a kernel."
        )

    def __setitem__(self, idx_obj, val):
        """Assigns a new value to the position represented by idx_obj in
        the self._data ndarray.
        """

        raise NotImplementedError(
            "The data of a LocalAccessor object can only be accessed "
            "inside a kernel."
        )


class _LocalAccessorMock:
    """Mock class that is used to represent a local accessor inside a "kernel".

    A LocalAccessor represents a device-only memory allocation and the
    class is designed in a way to not have any data container backing up the
    actual memory storage. Instead, the _LocalAccessorMock class is used to
    represent a local_accessor that has an actual numpy ndarray backing it up.
    Whenever, a LocalAccessor object is passed to `func`:kernel_api.call_kernel`
    it is converted to a _LocalAccessor internally. That way the data and
    access function on the data only works inside a kernel to simulate
    device-only memory allocation and outside the kernel the data for a
    LocalAccessor is not accessible.
    """

    def __init__(self, local_accessor: LocalAccessor):
        self._data = numpy.empty(
            local_accessor._shape, dtype=local_accessor._dtype
        )

    def __getitem__(self, idx_obj):
        """Returns the value stored at the position represented by idx_obj in
        the self._data ndarray.
        """

        return self._data[idx_obj]

    def __setitem__(self, idx_obj, val):
        """Assigns a new value to the position represented by idx_obj in
        the self._data ndarray.
        """

        self._data[idx_obj] = val
