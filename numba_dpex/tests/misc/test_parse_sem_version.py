# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba_dpex import parse_sem_version


class TestParseSemVersion:
    def test_parse_sem_version(self):
        assert parse_sem_version("0.56.4") == (0, 56, 4)
        assert parse_sem_version("0.57.0") == (0, 57, 0)
        assert parse_sem_version("0.57.0rc1") == (0, 57, 0)
        assert parse_sem_version("0.58.1dev0") == (0, 58, 1)
