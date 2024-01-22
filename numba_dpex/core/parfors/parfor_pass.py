# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
This module follows the logic of numba/parfors/parfor.py with changes required
to use it with dpnp instead of numpy.
"""

import warnings

from numba.core import config, errors, ir, types
from numba.core.compiler_machinery import register_pass
from numba.core.ir_utils import (
    dprint_func_ir,
    mk_alloc,
    mk_unique_var,
    next_label,
)
from numba.core.typed_passes import ParforPass as NumpyParforPass
from numba.core.typed_passes import _reload_parfors
from numba.parfors.parfor import (
    ConvertInplaceBinop,
    ConvertLoopPass,
    ConvertNumpyPass,
    ConvertReducePass,
    ConvertSetItemPass,
    Parfor,
)
from numba.parfors.parfor import ParforPass as _NumpyParforPass
from numba.parfors.parfor import (
    _make_index_var,
    _mk_parfor_loops,
    repr_arrayexpr,
    signature,
)
from numba.stencils.stencilparfor import StencilPass

from numba_dpex.numba_patches.patch_arrayexpr_tree_to_ir import (
    _arrayexpr_tree_to_ir,
)


class ConvertDPNPPass(ConvertNumpyPass):
    """
    Convert supported Dpnp functions, as well as arrayexpr nodes, to
    parfor nodes.

    Based on the ConvertNumpyPass. Lot's of code was copy-pasted, with minor
    changes due to lack of extensibility of the original package.
    """

    def __init__(self, pass_states):
        super().__init__(pass_states)

    def _arrayexpr_to_parfor(self, equiv_set, lhs, arrayexpr, avail_vars):
        """generate parfor from arrayexpr node, which is essentially a
        map with recursive tree.

        Exactly same as the original one, but with mock to _arrayexpr_tree_to_ir
        """
        pass_states = self.pass_states
        scope = lhs.scope
        loc = lhs.loc
        expr = arrayexpr.expr
        arr_typ = pass_states.typemap[lhs.name]
        el_typ = arr_typ.dtype

        # generate loopnests and size variables from lhs correlations
        size_vars = equiv_set.get_shape(lhs)
        index_vars, loopnests = _mk_parfor_loops(
            pass_states.typemap, size_vars, scope, loc
        )

        # generate init block and body
        init_block = ir.Block(scope, loc)
        init_block.body = mk_alloc(
            pass_states.typingctx,
            pass_states.typemap,
            pass_states.calltypes,
            lhs,
            tuple(size_vars),
            el_typ,
            scope,
            loc,
            pass_states.typemap[lhs.name],
        )
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var("$expr_out_var"), loc)
        pass_states.typemap[expr_out_var.name] = el_typ

        index_var, index_var_typ = _make_index_var(
            pass_states.typemap, scope, index_vars, body_block
        )

        body_block.body.extend(
            _arrayexpr_tree_to_ir(
                pass_states.func_ir,
                pass_states.typingctx,
                pass_states.typemap,
                pass_states.calltypes,
                equiv_set,
                init_block,
                expr_out_var,
                expr,
                index_var,
                index_vars,
                avail_vars,
            )
        )

        pat = ("array expression {}".format(repr_arrayexpr(arrayexpr.expr)),)

        parfor = Parfor(
            loopnests,
            init_block,
            {},
            loc,
            index_var,
            equiv_set,
            pat[0],
            pass_states.flags,
        )

        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        pass_states.calltypes[setitem_node] = signature(
            types.none, pass_states.typemap[lhs.name], index_var_typ, el_typ
        )
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT >= 1:
            print("parfor from arrayexpr")
            parfor.dump()
        return parfor


class _ParforPass(_NumpyParforPass):
    """ParforPass class is responsible for converting NumPy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.

    Based on the _NumpyParforPass. Lot's of code was copy-pasted, with minor
    changes due to lack of extensibility of the original package.
    """

    def run(self):
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible and optimize the IR.

        Exactly same as the original one, but with mock ConvertNumpyPass to
        ConvertDPNPPass.
        """
        self._pre_run()
        # run stencil translation to parfor
        if self.options.stencil:
            stencil_pass = StencilPass(
                self.func_ir,
                self.typemap,
                self.calltypes,
                self.array_analysis,
                self.typingctx,
                self.targetctx,
                self.flags,
            )
            stencil_pass.run()
        if self.options.setitem:
            ConvertSetItemPass(self).run(self.func_ir.blocks)
        if self.options.numpy:
            ConvertDPNPPass(self).run(self.func_ir.blocks)
        if self.options.reduction:
            ConvertReducePass(self).run(self.func_ir.blocks)
        if self.options.prange:
            ConvertLoopPass(self).run(self.func_ir.blocks)
        if self.options.inplace_binop:
            ConvertInplaceBinop(self).run(self.func_ir.blocks)

        # setup diagnostics now parfors are found
        self.diagnostics.setup(self.func_ir, self.options.fusion)

        dprint_func_ir(self.func_ir, "after parfor pass")


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforPass(NumpyParforPass):
    """Based on the NumpyParforPass. Lot's of code was copy-pasted, with minor
    changes due to lack of extensibility of the original package.
    """

    _name = "dpnp_parfor_pass"

    def __init__(self):
        NumpyParforPass.__init__(self)

    def run_pass(self, state):
        """
        Convert data-parallel computations into Parfor nodes.

        Exactly same as the original one, but with mock to _ParforPass.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _ParforPass(
            state.func_ir,
            state.typemap,
            state.calltypes,
            state.return_type,
            state.typingctx,
            state.targetctx,
            state.flags.auto_parallel,
            state.flags,
            state.metadata,
            state.parfor_diagnostics,
        )
        parfor_pass.run()

        # check the parfor pass worked and warn if it didn't
        has_parfor = False
        for blk in state.func_ir.blocks.values():
            for stmnt in blk.body:
                if isinstance(stmnt, Parfor):
                    has_parfor = True
                    break
            else:
                continue
            break

        if not has_parfor:
            # parfor calls the compiler chain again with a string
            if not (
                config.DISABLE_PERFORMANCE_WARNINGS
                or state.func_ir.loc.filename == "<string>"
            ):
                url = (
                    "https://numba.readthedocs.io/en/stable/user/"
                    "parallel.html#diagnostics"
                )
                msg = (
                    "\nThe keyword argument 'parallel=True' was specified "
                    "but no transformation for parallel execution was "
                    "possible.\n\nTo find out why, try turning on parallel "
                    "diagnostics, see %s for help." % url
                )
                warnings.warn(
                    errors.NumbaPerformanceWarning(msg, state.func_ir.loc)
                )

        # Add reload function to initialize the parallel backend.
        state.reload_init.append(_reload_parfors)
        return True
