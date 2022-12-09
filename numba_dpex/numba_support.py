# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb

numba_version = tuple(map(int, nb.__version__.split(".")[:2]))
