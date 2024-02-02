# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Definition of a new Literal type in numba-dpex that allows treating IntEnum
members as integer literals inside a JIT compiled function.
"""
from enum import IntEnum

from numba.core.pythonapi import box
from numba.core.typeconv import Conversion
from numba.core.types import Integer, Literal
from numba.core.typing.typeof import typeof

from numba_dpex.core.exceptions import IllegalIntEnumLiteralValueError
from numba_dpex.kernel_api.flag_enum import FlagEnum


class IntEnumLiteral(Literal, Integer):
    """A Literal type for IntEnum objects. The type contains the original Python
    value of the IntEnum class in it.
    """

    #  pylint: disable=W0231
    def __init__(self, value):
        self._literal_init(value)
        self.name = f"Literal[IntEnum]({value})"
        if issubclass(value, FlagEnum):
            basetype = typeof(value.basetype())
            Integer.__init__(
                self,
                name=self.name,
                bitwidth=basetype.bitwidth,
                signed=basetype.signed,
            )
        else:
            raise IllegalIntEnumLiteralValueError

    def can_convert_to(self, typingctx, other) -> bool:
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, Conversion.promote)
        return False


Literal.ctor_map[IntEnum] = IntEnumLiteral


@box(IntEnumLiteral)
def box_literal_integer(typ, val, ctx):
    """Defines how a Numba representation for an IntEnumLiteral object should
    be converted to a PyObject* object and returned back to Python.
    """
    val = ctx.context.cast(ctx.builder, val, typ, typ.literal_type)
    return ctx.box(typ.literal_type, val)
