# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from llvmlite import ir as llvmir
from numba.core import cgutils, errors, types
from numba.core.datamodel import default_manager
from numba.extending import intrinsic, overload

# can't import name because of the circular import
DPEX_TARGET_NAME = "dpex"


class Range(tuple):
    """A data structure to encapsulate a single kernel launch parameter.

    The range is an abstraction that describes the number of elements
    in each dimension of buffers and index spaces. It can contain
    1, 2, or 3 numbers, depending on the dimensionality of the
    object it describes.

    This is just a wrapper class on top of a 3-tuple. The kernel launch
    parameter is consisted of three int's. This class basically mimics
    the behavior of `sycl::range`.
    """

    UNDEFINED_DIMENSION = -1

    def __new__(cls, dim0, dim1=None, dim2=None):
        """Constructs a 1, 2, or 3 dimensional range.

        Args:
            dim0 (int): The range of the first dimension.
            dim1 (int, optional): The range of second dimension.
                                    Defaults to None.
            dim2 (int, optional): The range of the third dimension.
                                    Defaults to None.

        Raises:
            TypeError: If dim0 is not an int.
            TypeError: If dim1 is not an int.
            TypeError: If dim2 is not an int.
        """
        if not isinstance(dim0, int):
            raise TypeError("dim0 of a Range must be an int.")
        _values = [dim0]
        if dim1:
            if not isinstance(dim1, int):
                raise TypeError("dim1 of a Range must be an int.")
            _values.append(dim1)
            if dim2:
                if not isinstance(dim2, int):
                    raise TypeError("dim2 of a Range must be an int.")
                _values.append(dim2)
        return super(Range, cls).__new__(cls, tuple(_values))

    def get(self, index):
        """Returns the range of a single dimension.

        Args:
            index (int): The index of the dimension, i.e. [0,2]

        Returns:
            int: The range of the dimension indexed by `index`.
        """
        return self[index]

    def size(self):
        """Returns the size of a range.

        Returns the size of a range by multiplying
        the range of the individual dimensions.

        Returns:
            int: The size of a range.
        """
        n = len(self)
        if n > 2:
            return self[0] * self[1] * self[2]
        elif n > 1:
            return self[0] * self[1]
        else:
            return self[0]

    @property
    def ndim(self) -> int:
        """Returns the rank of a Range object.

        Returns:
            int: Number of dimensions in the Range object
        """
        return len(self)

    @property
    def dim0(self) -> int:
        """Return the extent of the first dimension for the Range object.

        Returns:
            int: Extent of first dimension for the Range object
        """
        return self[0]

    @property
    def dim1(self) -> int:
        """Return the extent of the second dimension for the Range object.

        Returns:
            int: Extent of second dimension for the Range object or -1 for 1D
            Range
        """
        try:
            return self[1]
        except IndexError:
            return Range.UNDEFINED_DIMENSION

    @property
    def dim2(self) -> int:
        """Return the extent of the second dimension for the Range object.

        Returns:
            int: Extent of second dimension for the Range object or -1 for 1D or
            2D Range
        """
        try:
            return self[2]
        except IndexError:
            return Range.UNDEFINED_DIMENSION


class NdRange:
    """A class to encapsulate all kernel launch parameters.

    The NdRange defines the index space for a work group as well as
    the global index space. It is passed to parallel_for to execute
    a kernel on a set of work items.

    This class basically contains two Range object, one for the global_range
    and the other for the local_range. The global_range parameter contains
    the global index space and the local_range parameter contains the index
    space of a work group. This class mimics the behavior of `sycl::nd_range`
    class.
    """

    def __init__(self, global_size, local_size):
        """Constructor for NdRange class.

        Args:
            global_size (Range or tuple of int's): The values for
                the global_range.
            local_size (Range or tuple of int's, optional): The values for
                the local_range. Defaults to None.
        """
        if isinstance(global_size, Range):
            self._global_range = global_size
        elif isinstance(global_size, Iterable):
            self._global_range = Range(*global_size)
        else:
            raise TypeError(
                "Unknown argument type for NdRange global_size, "
                + "must be of either type Range or Iterable of int's."
            )

        if isinstance(local_size, Range):
            self._local_range = local_size
        elif isinstance(local_size, Iterable):
            self._local_range = Range(*local_size)
        else:
            raise TypeError(
                "Unknown argument type for NdRange local_size, "
                + "must be of either type Range or Iterable of int's."
            )

    @property
    def global_range(self):
        """Accessor for global_range.

        Returns:
            Range: The `global_range` `Range` object.
        """
        return self._global_range

    @property
    def local_range(self):
        """Accessor for local_range.

        Returns:
            Range: The `local_range` `Range` object.
        """
        return self._local_range

    def get_global_range(self):
        """Returns a Range defining the index space.

        Returns:
            Range: A `Range` object defining the index space.
        """
        return self._global_range

    def get_local_range(self):
        """Returns a Range defining the index space of a work group.

        Returns:
            Range: A `Range` object to specify index space of a work group.
        """
        return self._local_range

    def __str__(self):
        """str() function for NdRange class.

        Returns:
            str: str representation for NdRange class.
        """
        return (
            "(" + str(self._global_range) + ", " + str(self._local_range) + ")"
        )

    def __repr__(self):
        """repr() function for NdRange class.

        Returns:
            str: str representation for NdRange class.
        """
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, NdRange):
            return (
                self.global_range == other.global_range
                and self.local_range == other.local_range
            )
        else:
            return False


@intrinsic(target=DPEX_TARGET_NAME)
def _intrin_range_alloc(typingctx, ty_dim0, ty_dim1, ty_dim2, ty_range):
    ty_retty = ty_range.instance_type
    sig = ty_retty(
        ty_dim0,
        ty_dim1,
        ty_dim2,
        ty_range,
    )

    def codegen(context, builder, sig, args):
        typ = sig.return_type
        dim0, dim1, dim2, _ = args
        range_struct = cgutils.create_struct_proxy(typ)(context, builder)
        range_struct.dim0 = dim0

        if not isinstance(sig.args[1], types.NoneType):
            range_struct.dim1 = dim1
        else:
            range_struct.dim1 = llvmir.Constant(
                llvmir.types.IntType(64), Range.UNDEFINED_DIMENSION
            )

        if not isinstance(sig.args[2], types.NoneType):
            range_struct.dim2 = dim2
        else:
            range_struct.dim2 = llvmir.Constant(
                llvmir.types.IntType(64), Range.UNDEFINED_DIMENSION
            )

        range_struct.ndim = llvmir.Constant(llvmir.types.IntType(64), typ.ndim)

        return range_struct._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def _intrin_ndrange_alloc(
    typingctx, ty_global_range, ty_local_range, ty_ndrange
):
    ty_retty = ty_ndrange.instance_type
    sig = ty_retty(
        ty_global_range,
        ty_local_range,
        ty_ndrange,
    )
    range_datamodel = default_manager.lookup(ty_global_range)

    def codegen(context, builder, sig, args):
        typ = sig.return_type

        global_range, local_range, _ = args
        ndrange_struct = cgutils.create_struct_proxy(typ)(context, builder)
        ndrange_struct.ndim = llvmir.Constant(
            llvmir.types.IntType(64), typ.ndim
        )
        ndrange_struct.gdim0 = builder.extract_value(
            global_range,
            range_datamodel.get_field_position("dim0"),
        )
        ndrange_struct.gdim1 = builder.extract_value(
            global_range,
            range_datamodel.get_field_position("dim1"),
        )
        ndrange_struct.gdim2 = builder.extract_value(
            global_range,
            range_datamodel.get_field_position("dim2"),
        )
        ndrange_struct.ldim0 = builder.extract_value(
            local_range,
            range_datamodel.get_field_position("dim0"),
        )
        ndrange_struct.ldim1 = builder.extract_value(
            local_range,
            range_datamodel.get_field_position("dim1"),
        )
        ndrange_struct.ldim2 = builder.extract_value(
            local_range,
            range_datamodel.get_field_position("dim2"),
        )

        return ndrange_struct._getvalue()

    return sig, codegen


@overload(Range, target=DPEX_TARGET_NAME)
def _ol_range_init(dim0, dim1=None, dim2=None):
    """Numba overload of the Range constructor to make it usable inside an
    njit and dpjit decorated function.

    """
    from numba_dpex.core.types import RangeType

    ndims = 1
    ty_optional_dims = (dim1, dim2)

    # A Range should at least have the 0th dimension populated
    if not isinstance(dim0, types.Integer):
        raise errors.TypingError(
            "Expected a Range's dimension should to be an Integer value, but "
            "encountered " + dim0.name
        )

    for ty_dim in ty_optional_dims:
        if isinstance(ty_dim, types.Integer):
            ndims += 1
        elif ty_dim is not None:
            raise errors.TypingError(
                "Expected a Range's dimension to be an Integer value, "
                f"but {type(ty_dim)} was provided."
            )

    ret_ty = RangeType(ndims)

    def impl(dim0, dim1=None, dim2=None):
        return _intrin_range_alloc(dim0, dim1, dim2, ret_ty)

    return impl


@overload(NdRange, target=DPEX_TARGET_NAME)
def _ol_ndrange_init(global_range, local_range):
    """Numba overload of the NdRange constructor to make it usable inside an
    njit and dpjit decorated function.

    """
    from numba_dpex.core.exceptions import UnmatchedNumberOfRangeDimsError
    from numba_dpex.core.types import NdRangeType, RangeType

    if not isinstance(global_range, RangeType):
        raise errors.TypingError(
            "Only global range values specified as a Range are "
            "supported inside dpjit"
        )

    if not isinstance(local_range, RangeType):
        raise errors.TypingError(
            "Only local range values specified as a Range are "
            "supported inside dpjit"
        )

    if not global_range.ndim == local_range.ndim:
        raise UnmatchedNumberOfRangeDimsError(
            kernel_name="",
            global_ndims=global_range.ndim,
            local_ndims=local_range.ndim,
        )

    ret_ty = NdRangeType(global_range.ndim)

    def impl(global_range, local_range):
        return _intrin_ndrange_alloc(global_range, local_range, ret_ty)

    return impl
