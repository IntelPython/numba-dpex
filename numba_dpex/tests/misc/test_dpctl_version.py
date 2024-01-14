# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl

from numba_dpex import dpctl_sem_version


def test_dpctl_version():
    dpctl_v = dpctl.__version__
    computed_v = ".".join(str(n) for n in dpctl_sem_version)
    n = len(computed_v)
    assert n <= len(dpctl_v)
    assert computed_v == dpctl_v[:n]
