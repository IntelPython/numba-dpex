# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.types import Boolean, Float, Integer, NoneType

# Short names for numba types supported in dpex kernel

none = NoneType("none")

boolean = bool_ = Boolean("bool")

uint32 = Integer("uint32")
uint64 = Integer("uint64")
int32 = Integer("int32")
int64 = Integer("int64")
float32 = Float("float32")
float64 = Float("float64")


# Aliases to NumPy type names

b1 = bool_
i4 = int32
i8 = int64
u4 = uint32
u8 = uint64
f4 = float32
f8 = float64

float_ = float32
double = float64
void = none
