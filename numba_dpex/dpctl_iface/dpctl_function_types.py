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

from numba import types
from numba.core.typing import signature


def dpctl_get_current_queue():
    ret_type = types.voidptr
    sig = signature(ret_type)
    return types.ExternalFunction("DPCTLQueueMgr_GetCurrentQueue", sig)


def dpctl_malloc_shared():
    ret_type = types.voidptr
    sig = signature(ret_type, types.int64, types.voidptr)
    return types.ExternalFunction("DPCTLmalloc_shared", sig)


def dpctl_queue_memcpy():
    ret_type = types.voidptr
    args = types.voidptr, types.voidptr, types.voidptr, types.int64
    sig = signature(ret_type, *args)
    return types.ExternalFunction("DPCTLQueue_Memcpy", sig)


def dpctl_event_wait():
    ret_type = types.voidptr
    sig = signature(ret_type, types.voidptr)
    return types.ExternalFunction("DPCTLEvent_Wait", sig)


def dpctl_event_delete():
    ret_type = types.void
    sig = signature(ret_type, types.voidptr)
    return types.ExternalFunction("DPCTLEvent_Delete", sig)


def dpctl_free_with_queue():
    ret_type = types.void
    sig = signature(ret_type, types.voidptr, types.voidptr)
    return types.ExternalFunction("DPCTLfree_with_queue", sig)


def dpctl_queue_wait():
    ret_type = types.void
    sig = signature(ret_type, types.voidptr)
    return types.ExternalFunction("DPCTLQueue_Wait", sig)
