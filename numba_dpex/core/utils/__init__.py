# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from .suai_helper import SyclUSMArrayInterface, get_info_from_suai

__all__ = [
    "get_info_from_suai",
    "SyclUSMArrayInterface",
]
