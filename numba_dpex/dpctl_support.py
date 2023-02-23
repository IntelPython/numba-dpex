# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl


def _parse_version():
    t = dpctl.__version__.split(".")
    if len(t) > 1:
        try:
            return tuple(map(int, t[:2]))
        except ValueError:
            return (0, 0)
    else:
        return (0, 0)


dpctl_version = _parse_version()

del _parse_version
