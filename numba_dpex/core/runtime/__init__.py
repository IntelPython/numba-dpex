# SPDX-FileCopyrightText: 2021 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import llvmlite.binding as ll

from ._dpexrt_python import c_helpers

# Register the helper function in _dpexrt_python so that we can insert
# calls to them via llvmlite.
for (
    py_name,
    c_address,
) in c_helpers.items():
    ll.add_symbol(py_name, c_address)
