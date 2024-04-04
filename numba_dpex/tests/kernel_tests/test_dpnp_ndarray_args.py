# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

import numba_dpex as ndpx
from numba_dpex import float32
from numba_dpex import kernel_api as kapi

COEFFICIENTS = dpnp.asarray(
    [
        [
            -0.008086340206185568,
            0.242590206185567,
            -0.6172680412371134,
            0,
        ],
        [
            0.015431701030927836,
            -1.168492268041237,
            27.60438144329897,
            -188.1443298969072,
        ],
        [
            -0.0036404639175257733,
            0.5480025773195877,
            -23.890463917525775,
            326.8041237113402,
        ],
        [
            -0.010869845360824743,
            1.4155283505154639,
            -58.59149484536083,
            789.4845360824743,
        ],
        [
            -0.0028801546391752576,
            0.21707474226804124,
            1.3311855670103092,
            -209.22680412371133,
        ],
        [
            0.042390463917525774,
            -7.9316365979381445,
            490.25386597938143,
            -9987.680412371134,
        ],
        [
            -0.061681701030927835,
            13.923518041237113,
            -1039.6069587628865,
            25709.072164948455,
        ],
        [
            0.029336340206185568,
            -7.920811855670103,
            707.9394329896908,
            -20892.164948453606,
        ],
    ],
    dtype=dpnp.float32,
)


@ndpx.kernel()
def _kernel(nditem: kapi.NdItem, coefficients):
    c = kapi.PrivateArray(4, dtype=float32)
    gr: kapi.Group = nditem.get_group()
    gr_id = gr.get_group_id(0)

    c[0] = coefficients[gr_id][0]
    c[1] = coefficients[gr_id][1]
    c[2] = coefficients[gr_id][2]
    c[3] = coefficients[gr_id][3]


def test_setting_private_from_dpnp_ndarray():
    N_SEGMENTS = 8
    LOCAL_SIZE = 128
    # Number of points in the batch. Each work item processes one batch
    N_POINTS_PER_WORK_ITEM = 4
    # Number of points processed by a work group
    N_POINTS_PER_WORK_GROUP = N_POINTS_PER_WORK_ITEM * LOCAL_SIZE
    # Total number of points
    N_POINTS = N_POINTS_PER_WORK_GROUP * N_SEGMENTS
    global_range = ndpx.Range(N_POINTS // N_POINTS_PER_WORK_ITEM)
    local_range = ndpx.Range(LOCAL_SIZE)
    try:
        ndpx.call_kernel(
            _kernel, ndpx.NdRange(global_range, local_range), COEFFICIENTS
        )
    except Exception as e:
        assert (
            False
        ), f"'_kernel' raised an exception {e} when passing a dpnp.ndarray arg"
