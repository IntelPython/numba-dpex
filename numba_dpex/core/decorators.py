# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from warnings import warn

from numba.core import decorators, sigutils, typeinfer
from numba.core.target_extension import (
    jit_registry,
    resolve_dispatcher_from_str,
    target_registry,
)

from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME
from numba_dpex.kernel_api_impl.spirv.dispatcher import SPIRVKernelDispatcher
from numba_dpex.kernel_api_impl.spirv.target import (
    SPIRV_TARGET_NAME,
    CompilationMode,
)


def _parse_func_or_sig(signature_or_function):
    # Handle signature (borrowed from numba). swapped signature and list check
    if signature_or_function is None:
        # No signature, no function
        pyfunc = None
        sigs = []
    elif sigutils.is_signature(signature_or_function):
        # A single signature is passed
        pyfunc = None
        sigs = [signature_or_function]
    elif isinstance(signature_or_function, list):
        # A list of signatures is passed
        pyfunc = None
        sigs = signature_or_function
    else:
        # A function is passed
        pyfunc = signature_or_function
        sigs = []

    return pyfunc, sigs


def kernel(function_or_signature=None, **options):
    """A decorator to compile a function written using :py:mod:`numba_dpex.kernel_api`.

    The ``kernel`` decorator triggers the compilation of a function written
    using the data-parallel kernel programming API exposed by
    :py:mod:`numba_dpex.kernel_api`. Such a function is conceptually
    equivalent to a kernel function written in the C++ SYCL eDSL. The
    decorator will compile the function based on the types of the arguments
    to a SPIR-V binary that can be executed either on OpenCL CPU, GPU
    devices or Intel Level Zero GPU devices.

    Any function to be compilable using the kernel decorator should
    adhere to the following semantic rules:

    - The first argument to the function should be either an instance of the
      :class:`numba_dpex.kernel_api.Item` class or an instance of the
      :class:`numba_dpex.kernel_api.NdItem`.

    - The function should not return any value.

    - The function should have at least one array type argument that can
      either be an instance of ``dpnp.ndarray`` or an instance of
      ``dpctl.tensor.usm_ndarray``.


    Args:
        signature_or_function (optional): An optional signature or list of
            signatures for which a function is to be compiled. Passing in a
            signature "specializes" the decorated function and no other versions
            of the function will be compiled. A function can also be
            directly passed instead of a signature and the signature will get
            inferred from the function. The actual compilation happens on every
            invocation of the :func:`numba_dpex.experimental.call_kernel`
            function where the decorated function is passed in as an argument
            along with the argument values for the decorated function.
        options (optional):
            - **debug** (bool): Whether the compilation should happen in debug
              mode. *(Default = False)*
            - **inline_threshold** (int): Specifies the level of inlining that
              the compiler should attempt. *(Default = 2)*
    Returns:
        An instance of
        :class:`numba_dpex.kernel_api_impl.spirv.dispatcher.KernelDispatcher`.
        The ``KernelDispatcher`` object compiles the decorated function when
        passed in to :func:`numba_dpex.experimental.call_kernel`.

    Examples:

    1. Decorate a function and pass it to ``call_kernel`` for compilation and
       execution.

    .. code-block:: python

        import dpnp
        import numba_dpex as dpex
        from numba_dpex import kernel_api as kapi


        # Data parallel kernel implementing vector sum
        @dpex.kernel
        def vecadd(item: kapi.Item, a, b, c):
            i = item.get_id(0)
            c[i] = a[i] + b[i]


        N = 1024
        a = dpnp.ones(N)
        b = dpnp.ones_like(a)
        c = dpnp.zeros_like(a)
        dpex.call_kernel(vecadd, kapi.Range(N), a, b, c)

    2. Specializes a kernel and then compiles it directly before executing it
       via ``call_kernel``. The kernel is specialized to expect a 1-D
       ``dpnp.ndarray`` with either ``float32`` type elements or ``int64`` type
       elements.

    .. code-block:: python

        import dpnp
        import numba_dpex as dpex
        from numba_dpex import kernel_api as kapi
        from numba_dpex import DpnpNdArray, float32, int64
        from numba_dpex.core.types.kernel_api.index_space_ids import ItemType

        i64arrty = DpnpNdArray(ndim=1, dtype=int64, layout="C")
        f32arrty = DpnpNdArray(ndim=1, dtype=float32, layout="C")
        item_ty = ItemType(ndim=1)

        specialized_kernel = dpex.kernel(
            [
                (item_ty, i64arrty, i64arrty, i64arrty),
                (item_ty, f32arrty, f32arrty, f32arrty),
            ]
        )


        def vecadd(item: kapi.Item, a, b, c):
            i = item.get_id(0)
            c[i] = a[i] + b[i]


        # Compile all specializations for vecadd
        precompiled_kernels = specialized_kernel(vecadd)
        N = 1024
        a = dpnp.ones(N, dtype=dpnp.int64)
        b = dpnp.ones_like(a)
        c = dpnp.zeros_like(a)
        # Call a specific pre-compiled version of vecadd
        dpex.call_kernel(precompiled_kernels, kapi.Range(N), a, b, c)

    """

    # dispatcher is a type:
    # <class 'numba_dpex.experimental.kernel_dispatcher.KernelDispatcher'>
    dispatcher = resolve_dispatcher_from_str(SPIRV_TARGET_NAME)
    if "_compilation_mode" in options:
        user_compilation_mode = options["_compilation_mode"]
        warn(
            "_compilation_mode is an internal flag that should not be set "
            "in the decorator. The decorator defined option "
            f"{user_compilation_mode} is going to be ignored."
        )
    options["_compilation_mode"] = CompilationMode.KERNEL

    # TODO: The options need to be evaluated and checked here like it is
    # done in numba.core.decorators.jit

    func, sigs = _parse_func_or_sig(function_or_signature)
    for sig in sigs:
        if isinstance(sig, str):
            raise NotImplementedError(
                "Specifying signatures as string is not yet supported"
            )

    def _kernel_dispatcher(pyfunc):
        disp: SPIRVKernelDispatcher = dispatcher(
            pyfunc=pyfunc,
            targetoptions=options,
        )

        if len(sigs) > 0:
            with typeinfer.register_dispatcher(disp):
                for sig in sigs:
                    disp.compile(sig)
                disp.disable_compile()

        return disp

    if func is None:
        return _kernel_dispatcher

    if not inspect.isfunction(func):
        raise ValueError(
            "Argument passed to the kernel decorator is neither a "
            "function object, nor a signature. If you are trying to "
            "specialize the kernel that takes a single argument, specify "
            "the return type as None explicitly."
        )
    return _kernel_dispatcher(func)


def device_func(function_or_signature=None, **options):
    """Compiles a device-callable function that can be only invoked from a kernel.

    The decorator is used to  express auxiliary device-only functions that can
    be called from a kernel or another device function, but are not callable
    from the host. This decorator :func:`numba_dpex.experimental.device_func`
    has no direct analogue in SYCL and primarily is provided to help programmers
    make their kapi applications modular.

    A ``device_func`` decorated function does not require the first argument to
    be a :class:`numba_dpex.kernel_api.Item` object or a
    :class:`numba_dpex.kernel_api.NdItem` object, and unlike a ``kernel``
    decorated function is allowed to return any value.
    All :py:mod:`numba_dpex.kernel_api` functionality can be used in a
    ``device_func`` decorated function.

    The decorator is also used to compile overloads in the ``DpexKernelTarget``.

    A ``device_func`` decorated function is not compiled down to device binary
    and instead is compiled down to LLVM IR. Final compilation to binary happens
    when the function is invoked from a ``kernel`` decorated function. The
    compilation happens this was to allow a ``device_func`` decorated function
    to be internally linked into the kernel module at the LLVM level, leading to
    more optimization opportunities.

    Args:
        signature_or_function (optional): An optional signature or list of
            signatures for which a function is to be compiled. Passing in a
            signature "specializes" the decorated function and no other versions
            of the function will be compiled. A function can also be
            directly passed instead of a signature and the signature will get
            inferred from the function. The actual compilation happens on every
            invocation of the decorated function from another ``device_func`` or
            ``kernel`` decorated function.
        options (optional):
            - **debug** (bool): Whether the compilation should happen in debug
              mode. *(Default = False)*
            - **inline_threshold** (int): Specifies the level of inlining that
              the compiler should attempt. *(Default = 2)*

    Returns:
        An instance of
        :class:`numba_dpex.kernel_api_impl.spirv.dispatcher.KernelDispatcher`.
        The ``KernelDispatcher`` object compiles the decorated function when
        it is called from another function.


    Example:

    .. code-block:: python

        import dpnp

        from numba_dpex import experimental as dpex_exp
        from numba_dpex import kernel_api as kapi


        @dpex_exp.device_func
        def increment_value(nd_item: NdItem, a):
            i = nd_item.get_global_id(0)

            a[i] += 1
            group_barrier(nd_item.get_group(), MemoryScope.DEVICE)

            if i == 0:
                for idx in range(1, a.size):
                    a[0] += a[idx]


        @dpex_exp.kernel
        def another_kernel(nd_item: NdItem, a):
            increment_value(nd_item, a)


        N = 16
        b = dpnp.ones(N, dtype=dpnp.int32)

        dpex_exp.call_kernel(another_kernel, dpex.NdRange((N,), (N,)), b)
    """
    dispatcher = resolve_dispatcher_from_str(SPIRV_TARGET_NAME)

    if "_compilation_mode" in options:
        user_compilation_mode = options["_compilation_mode"]
        warn(
            "_compilation_mode is an internal flag that should not be set "
            "in the decorator. The decorator defined option "
            f"{user_compilation_mode} is going to be ignored."
        )
    options["_compilation_mode"] = CompilationMode.DEVICE_FUNC

    func, sigs = _parse_func_or_sig(function_or_signature)
    for sig in sigs:
        if isinstance(sig, str):
            raise NotImplementedError(
                "Specifying signatures as string is not yet supported"
            )

    def _kernel_dispatcher(pyfunc):
        disp: SPIRVKernelDispatcher = dispatcher(
            pyfunc=pyfunc,
            targetoptions=options,
        )

        if len(sigs) > 0:
            with typeinfer.register_dispatcher(disp):
                for sig in sigs:
                    disp.compile(sig)
                disp.disable_compile()

        return disp

    if func is None:
        return _kernel_dispatcher

    return _kernel_dispatcher(function_or_signature)


# ----------------- Experimental dpjit decorator ------------------------------#


def dpjit(*args, **kws):
    if "nopython" in kws and kws["nopython"] is not True:
        warn("nopython is set for dpjit and is ignored", RuntimeWarning)
    if "forceobj" in kws:
        warn("forceobj is set for dpjit and is ignored", RuntimeWarning)
        del kws["forceobj"]
    if "pipeline_class" in kws:
        warn("pipeline class is set for dpjit and is ignored", RuntimeWarning)
        del kws["pipeline_class"]

    kws.update({"nopython": True})
    kws.update({"parallel": True})

    kws.update({"_target": DPEX_TARGET_NAME})

    return decorators.jit(*args, **kws)


# add it to the decorator registry, this is so e.g. @overload can look up a
# JIT function to do the compilation work.
jit_registry[target_registry[DPEX_TARGET_NAME]] = dpjit
jit_registry[target_registry[SPIRV_TARGET_NAME]] = device_func
