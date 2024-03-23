# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Demonstrates natural cubic spline implemented as an nd-range kernel.

Refer: https://en.wikipedia.org/wiki/Spline_interpolation
"""

import dpnp as np
from numba import float32
from numpy.testing import assert_almost_equal

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

# Interpolation domain
XLO = 10.0
XHI = 90.0

# Number of cubic polynomial segments
N_SEGMENTS = 8

LOCAL_SIZE = 128  # Number of batches
N_POINTS_PER_WORK_ITEM = (
    4  # Number of points in the batch. Each work item processes one batch
)
N_POINTS_PER_WORK_GROUP = (
    N_POINTS_PER_WORK_ITEM * LOCAL_SIZE
)  # Number of points processed by a work group
N_POINTS = N_POINTS_PER_WORK_GROUP * N_SEGMENTS  # Total number of points

# Natural cubic spline coefficients in interval [10, 90] with uniform grid
# Each work group processes its own segment
COEFFICIENTS = np.asarray(
    [
        [
            -0.008086340206185568,
            0.242590206185567,
            -0.6172680412371134,
            0,
        ],  # [10, 20]
        [
            0.015431701030927836,
            -1.168492268041237,
            27.60438144329897,
            -188.1443298969072,
        ],  # [20, 30]
        [
            -0.0036404639175257733,
            0.5480025773195877,
            -23.890463917525775,
            326.8041237113402,
        ],  # [30, 40]
        [
            -0.010869845360824743,
            1.4155283505154639,
            -58.59149484536083,
            789.4845360824743,
        ],  # [40, 50]
        [
            -0.0028801546391752576,
            0.21707474226804124,
            1.3311855670103092,
            -209.22680412371133,
        ],  # [50, 60]
        [
            0.042390463917525774,
            -7.9316365979381445,
            490.25386597938143,
            -9987.680412371134,
        ],  # [60, 70]
        [
            -0.061681701030927835,
            13.923518041237113,
            -1039.6069587628865,
            25709.072164948455,
        ],  # [70, 80]
        [
            0.029336340206185568,
            -7.920811855670103,
            707.9394329896908,
            -20892.164948453606,
        ],
    ],  # [80, 90]
    dtype=np.float32,  # We use single precision for interpolation
)


@dpex.kernel()
def kernel_polynomial(nditem: kapi.NdItem, x, y, coefficients):
    c = kapi.PrivateArray(
        4, dtype=float32
    )  # Coefficients of a polynomial of a given segment
    z = kapi.PrivateArray(1, dtype=float32)  # Keep x[i] in private memory

    gid = nditem.get_global_id(0)
    gr_id = nditem.get_group().get_group_id(0)

    # Polynomial coefficients are fixed within a workgroup
    c[0] = coefficients[gr_id][0]
    c[1] = coefficients[gr_id][1]
    c[2] = coefficients[gr_id][2]
    c[3] = coefficients[gr_id][3]

    # Each work item processes N_POINTS_PER_WORK_ITEM points
    for i in range(
        gid * N_POINTS_PER_WORK_ITEM, (gid + 1) * N_POINTS_PER_WORK_ITEM, 1
    ):
        z[0] = x[i]  # Copy current point into the private memory
        y[i] = ((c[0] * z[0] + c[1]) * z[0] + c[2]) * z[0] + c[
            3
        ]  # Coefficients are in the private memory too


def main():
    # Create arrays on the default device
    xp = np.arange(XLO, XHI, (XHI - XLO) / N_POINTS)
    yp = np.empty(xp.shape)

    print("Executing on device:")
    xp.device.print_device_info()
    global_range = kapi.Range(
        N_POINTS // N_POINTS_PER_WORK_ITEM,
    )
    local_range = kapi.Range(
        LOCAL_SIZE,
    )
    dpex.call_kernel(
        kernel_polynomial,
        dpex.NdRange(global_range, local_range),
        xp,
        yp,
        COEFFICIENTS,
    )

    # Copy results back to the host
    nyp = np.asnumpy(yp)
    # Basic check for correctness
    assert_almost_equal(nyp[2047], 39.97161865234375)

    print("Done...")


if __name__ == "__main__":
    main()
