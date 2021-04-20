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

from numba.core import types


def numba_type_to_dpctl_typenum(context, type):
    """
    This function looks up the dpctl defined enum values from
    ``DPCTLKernelArgType``.
    """

    val = None
    if type == types.int32 or isinstance(type, types.scalars.IntegerLiteral):
        # DPCTL_LONG_LONG
        val = context.get_constant(types.int32, 9)
    elif type == types.uint32:
        # DPCTL_UNSIGNED_LONG_LONG
        val = context.get_constant(types.int32, 10)
    elif type == types.boolean:
        # DPCTL_UNSIGNED_INT
        val = context.get_constant(types.int32, 5)
    elif type == types.int64:
        # DPCTL_LONG_LONG
        val = context.get_constant(types.int32, 9)
    elif type == types.uint64:
        # DPCTL_SIZE_T
        val = context.get_constant(types.int32, 11)
    elif type == types.float32:
        # DPCTL_FLOAT
        val = context.get_constant(types.int32, 12)
    elif type == types.float64:
        # DPCTL_DOUBLE
        val = context.get_constant(types.int32, 13)
    elif type == types.voidptr:
        # DPCTL_VOID_PTR
        val = context.get_constant(types.int32, 15)
    else:
        raise NotImplementedError

    return val
