# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple, Union

from numba.core import config
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import (
    add_offset_to_labels,
    get_name_var_table,
    get_unused_var_name,
    mk_unique_var,
    remove_dels,
    rename_labels,
    replace_var_names,
)


class ParforBodyArguments(NamedTuple):
    """
    Arguments containing information to inject parfor code inside kernel.
    """

    loop_body: any
    param_dict: any
    legal_loop_indices: any


def _print_block(block):
    for i, inst in enumerate(block.body):
        print("    ", i, inst)


def _print_body(body_dict):
    """Pretty-print a set of IR blocks."""
    for label, block in body_dict.items():
        print("label: ", label)
        _print_block(block)


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforSentinelReplacePass(FunctionPass):
    _name = "sentinel_inject"

    def __init__(self):
        FunctionPass.__init__(self)

    def _get_parfor_body_args(self, flags) -> Union[ParforBodyArguments, None]:
        if not hasattr(flags, "_parfor_body_args"):
            return None

        return flags._parfor_body_args

    def run_pass(self, state):
        flags = state["flags"]

        args = self._get_parfor_body_args(flags)

        if args is None:
            return True

        # beginning
        kernel_ir = state["func_ir"]
        loop_body = args.loop_body

        if config.DEBUG_ARRAY_OPT:
            print("kernel_ir dump ", type(kernel_ir))
            kernel_ir.dump()
            print("loop_body dump ", type(loop_body))
            _print_body(loop_body)

        # Determine the unique names of the scheduling and kernel functions.
        loop_body_var_table = get_name_var_table(loop_body)
        sentinel_name = get_unused_var_name("__sentinel__", loop_body_var_table)

        # rename all variables in kernel_ir afresh
        var_table = get_name_var_table(kernel_ir.blocks)
        new_var_dict = {}
        reserved_names = (
            [sentinel_name]
            + list(args.param_dict.values())
            + args.legal_loop_indices
        )
        for name, _ in var_table.items():
            if not (name in reserved_names):
                new_var_dict[name] = mk_unique_var(name)

        replace_var_names(kernel_ir.blocks, new_var_dict)

        kernel_stub_last_label = max(kernel_ir.blocks.keys()) + 1
        loop_body = add_offset_to_labels(loop_body, kernel_stub_last_label)

        # new label for splitting sentinel block
        new_label = max(loop_body.keys()) + 1

        from .kernel_builder import update_sentinel  # circular

        update_sentinel(kernel_ir, sentinel_name, loop_body, new_label)

        # FIXME: Why rename and remove dels causes the partial_sum array update
        # instructions to be removed.
        kernel_ir.blocks = rename_labels(kernel_ir.blocks)
        remove_dels(kernel_ir.blocks)

        if config.DEBUG_ARRAY_OPT:
            print("kernel_ir after remove dead")
            kernel_ir.dump()

        return True
