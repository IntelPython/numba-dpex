# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides string templates for numba_dpex.kernel decorated functions.

During lowering of a parfor node using the SPIRVKernelTarget, the node is
first converted into a kernel function. The module provides a set of templates
to generate the basic stub of a kernel function. The string template is
compiled down to Numba IR using the Numba compiler front end and then the
necessary body of the kernel function is inserted directly as Numba IR.
"""
