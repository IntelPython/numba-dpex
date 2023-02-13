# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from . import arrayobj


def ensure_dpnp(name):
    try:
        from . import dpnp_fptr_interface as dpnp_iface
    except ImportError:
        raise ImportError("dpnp is needed to call np.%s" % name)


def _init_dpnp():
    try:
        import os

        import dpnp

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.dirname(dpnp.__file__))
    except ImportError:
        pass


_init_dpnp()


DEBUG = None
