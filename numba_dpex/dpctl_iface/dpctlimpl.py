# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.imputils import Registry

registry = Registry("dpctlimpl")

lower_builtin = registry.lower
lower_getattr = registry.lower_getattr
lower_getattr_generic = registry.lower_getattr_generic
lower_setattr = registry.lower_setattr
lower_setattr_generic = registry.lower_setattr_generic
lower_cast = registry.lower_cast
lower_constant = registry.lower_constant
