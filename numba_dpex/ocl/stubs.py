# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

_stub_error = NotImplementedError("This is a stub.")

# mem fence
CLK_LOCAL_MEM_FENCE = 0x1
CLK_GLOBAL_MEM_FENCE = 0x2


def get_global_id(*args, **kargs):
    """
    OpenCL get_global_id()
    """
    raise _stub_error


def get_local_id(*args, **kargs):
    """
    OpenCL get_local_id()
    """
    raise _stub_error


def get_global_size(*args, **kargs):
    """
    OpenCL get_global_size()
    """
    raise _stub_error


def get_local_size(*args, **kargs):
    """
    OpenCL get_local_size()
    """
    raise _stub_error


def get_group_id(*args, **kargs):
    """
    OpenCL get_group_id()
    """
    raise _stub_error


def get_num_groups(*args, **kargs):
    """
    OpenCL get_num_groups()
    """
    raise _stub_error


def get_work_dim(*args, **kargs):
    """
    OpenCL get_work_dim()
    """
    raise _stub_error


def barrier(*args, **kargs):
    """
    OpenCL barrier()
    """
    raise _stub_error


def mem_fence(*args, **kargs):
    """
    OpenCL mem_fence()
    """
    raise _stub_error


def sub_group_barrier():
    """
    OpenCL 2.0 sub_group_barrier
    """
    raise _stub_error


class Stub(object):
    """A stub object to represent special objects that are meaningless
    outside the context of kernel compilation.
    """

    _description_ = "<dpex special value>"
    __slots__ = ()  # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)

    def __repr__(self):
        return self._description_


# -------------------------------------------------------------------------------
# private memory


class private(Stub):
    """private namespace"""

    _description_ = "<private>"

    def array(shape, dtype):
        """private.array(shape, dtype)

        Allocate a private array.
        """


# -------------------------------------------------------------------------------
# local memory


class local(Stub):
    """local namespace"""

    _description_ = "<local>"

    def array(shape, dtype):
        """local.array(shape, dtype)

        Allocate a local array.
        """


# -------------------------------------------------------------------------------
# atomic


class atomic(Stub):

    def add():
        """
        add(ary, idx, val)

        Performs atomic addition.

           Parameters:
               ary: An array on which the atomic addition is performed. Elements can only be of type
                    int32, int64, float32, or float64
               idx (int): Index of the array element ary[idx] on which atomic addition is performed
               val: The value of an increment, ary[idx] += val. The type must match the type of array elements, ary[]

           Returns:
                   The old value at the index location ary[idx] as if it is loaded atomically.

           Raises:
                Exception1: Description1
                Exception2: Description2
        """

    def sub():
        """
        sub(ary, idx, val)

        Performs atomic addition.

           Parameters:
               ary: An array on which the atomic subtraction is performed. Elements can only be of type
                    int32, int64, float32, or float64
               idx (int): Index of the array element ary[idx] on which atomic subtraction is performed
               val: The value of an increment, ary[idx] -= val. The type must match the type of array elements, ary[]

           Returns:
                   The old value at the index location ary[idx] as if it is loaded atomically.

           Raises:
                Exception1: Description1
                Exception2: Description2
        """
