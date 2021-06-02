# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numba.core import types, ir, typing

from numba_dppy.target import SPIR_LOCAL_ADDRSPACE
import numpy as np
import numba
from numba.np import numpy_support

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
    """A stub object to represent special objects which is meaningless
    outside the context of DPPY compilation context.
    """

    _description_ = "<dppy special value>"
    __slots__ = ()  # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)

    def __repr__(self):
        return self._description_


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
    """atomic namespace"""

    _description_ = "<atomic>"

    def add():
        """add(ary, idx, val)

        Perform atomic ary[idx] += val.

        Returns the old value at the index location as if it is loaded atomically.

        .. note:: Supported on int32, int64, float32, float64 operands only.
        """

    def sub():
        """sub(ary, idx, val)

        Perform atomic ary[idx] -= val.

        Returns the old value at the index location as if it is loaded atomically.

        .. note:: Supported on int32, int64, float32, float64 operands only.
        """
