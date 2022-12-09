#! /usr/bin/env python

# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest

dpnp = pytest.importorskip("dpnp")


def test_import_dpnp():
    """Test that import dpnp works"""
    import dpnp


def test_import_dpnp_fptr_interface():
    """Test that we can import dpnp_fptr_interface if dpnp is installed"""
    from numba_dpex.dpnp_iface import dpnp_fptr_interface
