# SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import ctypes as ct

import llvmlite.binding as ll

from . import _dpexrt_python
from ._dpexrt_python import c_helpers, get_queue_ref

# Register the helper function in _dpexrt_python so that we can insert
# calls to them via llvmlite.
for (
    py_name,
    c_address,
) in c_helpers.items():
    ll.add_symbol(py_name, c_address)

get_queue_ref_sig = ct.CFUNCTYPE(ct.c_void_p, ct.py_object)
get_queue_ref = ct.cast(get_queue_ref, get_queue_ref_sig)
