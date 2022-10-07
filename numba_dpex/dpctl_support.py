# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl

dpctl_version = tuple(map(int, dpctl.__version__.split(".")[:2]))
