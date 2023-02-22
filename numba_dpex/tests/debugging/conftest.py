#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def app():
    from .gdb import gdb

    return gdb()
