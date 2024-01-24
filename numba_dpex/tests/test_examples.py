#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os

import numba_dpex


def test_examples_available():
    package_path = os.path.dirname(numba_dpex.__file__)
    examples_path = os.path.join(package_path, "examples")

    assert os.path.isdir(examples_path)
