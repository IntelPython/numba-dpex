from numba.core import cgutils
from numba.core.callconv import MinimalCallConv


class DpexParforCallConv(MinimalCallConv):
    """Custom calling convention class used by numba-dpex.

    numba_dpex's calling convention derives from
    :class:`numba.core.callconv import MinimalCallConv`. The
    :class:`DpexCallConv` overrides :func:`call_function`.

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
    :class:`numba.core.callconv import MinimalCallConv`. The
    :class:`DpexCallConv` overrides :func:`call_function`.

    """

    def return_user_exc(
        self, builder, exc, exc_args=None, loc=None, func_name=None
    ):
        msg = "Python exceptions are unsupported in the CUDA C/C++ ABI"
        raise NotImplementedError(msg)

    def return_status_propagate(self, builder, status):
        msg = "Return status is unsupported in the CUDA C/C++ ABI"
        raise NotImplementedError(msg)
