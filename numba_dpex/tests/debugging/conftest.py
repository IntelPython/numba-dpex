#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from .gdb import gdb


@pytest.fixture
def app():
    g = gdb()

    yield g

    g.teardown_gdb()
