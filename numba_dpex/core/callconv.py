# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


from numba.core import cgutils
from numba.core.callconv import MinimalCallConv


class DpexParforCallConv(MinimalCallConv):
    """Custom calling convention class used by numba-dpex.

    numba_dpex's calling convention derives from
    :class:`numba.core.callconv.MinimalCallConv`. The
    :class:`DpexParforCallConv` overrides :func:`call_function`.

    Args:
        MinimalCallConv (numba.core.callconv.BaseCallConv): The base call
            convention class.
    """

    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """Call the Numba-compiled *callee*."""
        assert env is None
        retty = callee.args[0].type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(cgutils.get_null_value(retty), retvaltmp)
        arginfo = self.context.get_arg_packer(argtys)
        args = arginfo.as_arguments(builder, args)
        realargs = [retvaltmp] + list(args)
        code = builder.call(callee, realargs)
        status = self._get_return_status(builder, code)
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out


class DpexCallConv(DpexParforCallConv):
    """Custom calling convention class used by numba-dpex.

    numba_dpex's calling convention derives from
    :class:`DpexParforCallConv`. The :class:`DpexCallConv` overrides
    :func:`call_function` in the same way as in :class:`DpexParforCallConv`
    Except this class raises an exception when a `raise` or an `assert`
    statement is encountered in a `numba-dpex` kernel.

    Args:
        DpexParforCallConv (numba.core.callconv.MinimalCallConv): The minimal
            call convention class.
    """

    def return_user_exc(
        self, builder, exc, exc_args=None, loc=None, func_name=None
    ):
        msg = "Python exceptions and asserts are unsupported in numba-dpex kernel."
        raise NotImplementedError(msg)
