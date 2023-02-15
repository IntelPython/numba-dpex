from numba.core.lowering import Lower
from numba.parfors.parfor_lowering import (
    _lower_parfor_parallel as _lower_parfor_parallel_std,
)

from .parfor import Parfor


class ParforLower(Lower):
    """This is a custom lowering class that extends standard lowering so as
    to accommodate parfor.Parfor nodes."""

    # custom instruction lowering to handle parfor nodes
    def lower_inst(self, inst):
        if isinstance(inst, Parfor):
            if inst.lowerer is None:
                _lower_parfor_parallel_std(self, inst)
            else:
                inst.lowerer(self, inst)
        else:
            super().lower_inst(inst)
