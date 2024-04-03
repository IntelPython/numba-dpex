# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

from dpctl import get_device_cached_queue
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
from llvmlite.ir.types import DoubleType, FloatType
from numba import types
from numba.core import cgutils
from numba.core import config as numba_config
from numba.core.typing import signature
from numba.extending import intrinsic, overload_classmethod
from numba.np.arrayobj import (
    _parse_empty_args,
    _parse_empty_like_args,
    get_itemsize,
    make_array,
    populate_array,
)

from numba_dpex.core.runtime import context as dpexrt
from numba_dpex.core.types import DpnpNdArray
from numba_dpex.core.types.dpctl_types import DpctlSyclQueue
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl

# can't import name because of the circular import
DPEX_TARGET_NAME = "dpex"

_QueueRefPayload = namedtuple(
    "QueueRefPayload", ["queue_ref", "py_dpctl_sycl_queue_addr", "pyapi"]
)

_ArgTyAndValue = namedtuple("ArgTyAndValue", ["numba_ty", "llvmir_val"])


# XXX: The function should be moved into DpexTargetContext
def make_queue(context, builder, py_dpctl_sycl_queue):
    """Utility function used for allocating a new queue.

    This function will allocates a new queue (e.g. SYCL queue)
    during LLVM code generation (lowering). Given a target context,
    builder, array type, returns a LLVM value pointing at a numba-dpex
    runtime allocated queue.

    Args:
        context (numba.core.base.BaseContext): Any of the context
            derived from Numba's BaseContext
            (e.g. `numba.core.cpu.CPUContext`).
        builder (llvmlite.ir.builder.IRBuilder): The IR builder
            from `llvmlite` for code generation.
        py_dpctl_sycl_queue (dpctl.SyclQueue): A Python dpctl.SyclQueue object.

    Returns:
        ret (namedtuple): A namedtuple containing
        `llvmlite.ir.instructions.ExtractValue` as `queue_ref`,
        `llvmlite.ir.instructions.CastInstr` as `queue_address_ptr`
        and `numba.core.pythonapi.PythonAPI` as `pyapi`.
    """

    pyapi = context.get_python_api(builder)
    queue_struct_proxy = cgutils.create_struct_proxy(
        DpctlSyclQueue(py_dpctl_sycl_queue)
    )(context, builder)
    queue_struct_ptr = queue_struct_proxy._getpointer()
    queue_struct_voidptr = builder.bitcast(queue_struct_ptr, cgutils.voidptr_t)

    address = context.get_constant(types.intp, id(py_dpctl_sycl_queue))
    py_dpctl_sycl_queue_addr = builder.inttoptr(address, cgutils.voidptr_t)

    dpexrtCtx = dpexrt.DpexRTContext(context)
    dpexrtCtx.queuestruct_from_python(
        pyapi, py_dpctl_sycl_queue_addr, queue_struct_voidptr
    )

    queue_ref = queue_struct_proxy.queue_ref

    return_values = namedtuple(
        "return_values", "queue_ref queue_address_ptr pyapi"
    )
    ret = return_values(queue_ref, py_dpctl_sycl_queue_addr, pyapi)

    return ret


def _get_queue_ref(
    context,
    builder,
    returned_sycl_queue_ty,
    sycl_queue_arg: _ArgTyAndValue,
    array_arg: _ArgTyAndValue = None,
):
    """Returns an LLVM IR Value pointer to a DpctlSyclQueueRef

    The _get_queue_ref function is used by the intinsic functions that implement
    the overloads for dpnp array constructors: ``empty``, ``empty_like``,
    ``zeros``, ``zeros_like``, ``ones``, ``ones_like``, ``full``, ``full_like``.

    The function returns an LLVM IR Value corresponding to a dpctl.SyclQueue
    Python object's underlying ``_queue_ref`` pointer. If a non-None
    ``sycl_queue_arg`` is provided, then the ``_queue_ref`` attribute is
    extracted from the ``sycl_queue_arg``. If the ``sycl_queue_arg`` is
    None or omitted and an ``array_arg`` is provided, then the ``_queue_ref``
    is extracted from the unboxed representation of the ``array_arg``. If
    nether a non-None ``sycl_queue_arg`` nor an ``array_arg`` is provided,
    then a cached dpctl.SyclQueue is retreived from dpctl and unboxed on the fly
    and the ``_queue_ref`` from that unboxed queue is returned to caller.

    Args:
        context (numba.core.base.BaseContext): Any of the context
            derived from Numba's BaseContext
            (e.g. `numba.core.cpu.CPUContext`).
        builder (llvmlite.ir.builder.IRBuilder): The IR builder
            from `llvmlite` for code generation.
        returned_sycl_queue_ty: An instance of numba_dpex.types.DpctlSyclQueue
        sycl_queue_arg: A 2-tuple storing the numba inferred type and the
            corresponding LLVM IR value for a dpctl.SyclQueue Python object.
        array_arg: A 2-tuple storing the numba inferred type and the
            corresponding LLVM IR value for a dpnp.ndarray Python object.

    Return:
        A namedtuple wrapping the queue_ref pointer, an optional address to
        a dpctl.SyclQueue Python object, and an option instance of the python
        api wrapper in the CPUContext.

    """

    queue_ref = None
    py_dpctl_sycl_queue_addr = None
    pyapi = None

    if not isinstance(
        sycl_queue_arg.numba_ty, (types.misc.NoneType, types.misc.Omitted)
    ) and isinstance(sycl_queue_arg.numba_ty, DpctlSyclQueue):
        if not isinstance(
            sycl_queue_arg.llvmir_val.type, llvmir.LiteralStructType
        ):
            raise AssertionError(
                "Expected the queue_arg to be an llvmir.LiteralStructType"
            )
        sycl_queue_dm = context.data_model_manager.lookup(
            sycl_queue_arg.numba_ty
        )
        queue_ref = builder.extract_value(
            sycl_queue_arg.llvmir_val,
            sycl_queue_dm.get_field_position("queue_ref"),
        )
    elif array_arg is not None:
        dpnp_ndarray_dm = context.data_model_manager.lookup(array_arg.numba_ty)
        queue_ref = builder.extract_value(
            array_arg.llvmir_val,
            dpnp_ndarray_dm.get_field_position("sycl_queue"),
        )
    else:
        if not isinstance(sycl_queue_arg.llvmir_val.type, llvmir.PointerType):
            # TODO: check if the pointer is null
            raise AssertionError(
                "Expected the queue_arg to be an llvmir.PointerType"
            )
        # We are emulating queue boxing here to pass it as an argument.
        # TODO: do we have a better way?
        py_dpctl_sycl_queue = get_device_cached_queue(
            returned_sycl_queue_ty.sycl_device
        )
        # TODO: make sure that queue is not getting garbage collected between
        # lowering and executing.
        # cache prevent it from happening but it is not clean way
        (queue_ref, py_dpctl_sycl_queue_addr, pyapi) = make_queue(
            context, builder, py_dpctl_sycl_queue
        )

    ret = _QueueRefPayload(queue_ref, py_dpctl_sycl_queue_addr, pyapi)
    return ret


def _update_queue_attr(array, queue):
    """Assigns the sycl_queue member of an usmarystruct_t instance.

    After creating a new usmarystruct_t struct (e.g. in _empty_nd_impl) the
    members of the struct are populated by calling
    numba.np.arrayobj.populate_array. The populate_array function does not
    update the sycl_queue member as populate_array is written specifically for
    numba's arystruct_t type that does not have a sycl_queue member. The
    _update_queue_attr is a helper function to update the sycl_queue field.
    """

    attr = dict(sycl_queue=queue)
    for k, v in attr.items():
        setattr(array, k, v)


def _empty_nd_impl(context, builder, arrtype, shapes, queue_ref):
    """Utility function used for allocating a new array.

    This function is used for allocating a new array during LLVM code
    generation (lowering).  Given a target context, builder, array
    type, and a tuple or list of lowered dimension sizes, returns a
    LLVM value pointing at a Numba runtime allocated array.
    """

    arycls = make_array(arrtype)
    ary = arycls(context, builder)

    datatype = context.get_data_type(arrtype.dtype)
    itemsize = context.get_constant(types.intp, get_itemsize(context, arrtype))

    # compute array length
    arrlen = context.get_constant(types.intp, 1)
    overflow = Constant(llvmir.IntType(1), 0)
    for s in shapes:
        arrlen_mult = builder.smul_with_overflow(arrlen, s)
        arrlen = builder.extract_value(arrlen_mult, 0)
        overflow = builder.or_(overflow, builder.extract_value(arrlen_mult, 1))

    if arrtype.ndim == 0:
        strides = ()
    elif arrtype.layout == "C":
        strides = [itemsize]
        for dimension_size in reversed(shapes[1:]):
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(reversed(strides))
    elif arrtype.layout == "F":
        strides = [itemsize]
        for dimension_size in shapes[:-1]:
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(strides)
    else:
        raise NotImplementedError(
            "Don't know how to allocate array with layout '{0}'.".format(
                arrtype.layout
            )
        )

    # Check overflow, numpy also does this after checking order
    allocsize_mult = builder.smul_with_overflow(arrlen, itemsize)
    allocsize = builder.extract_value(allocsize_mult, 0)
    overflow = builder.or_(overflow, builder.extract_value(allocsize_mult, 1))

    with builder.if_then(overflow, likely=False):
        # Raise same error as numpy, see:
        # https://github.com/numpy/numpy/blob/2a488fe76a0f732dc418d03b452caace161673da/numpy/core/src/multiarray/ctors.c#L1095-L1101    # noqa: E501
        context.call_conv.return_user_exc(
            builder,
            ValueError,
            (
                "array is too big; `arr.size * arr.dtype.itemsize` is larger "
                "than the maximum possible size.",
            ),
        )

    # The passed in queue_ref if used to allocate a MemInfo object needs to be
    # copied first. The reason for the copy is to properly manage the lifetime
    # of the queue_ref object. The original object is owned by the parent
    # dpctl.SyclQueue object and is deleted when the dpctl.SyclQueue is garbage
    # collected. Whereas, the copied queue_ref is to be owned by the
    # NRT_External_Allocator object of MemInfo, and its lifetime is tied to the
    # MemInfo object.
    queue_ref_copy = sycl.dpctl_queue_copy(builder, queue_ref)

    usm_ty = arrtype.usm_type
    usm_ty_map = {"device": 1, "shared": 2, "host": 3}
    usm_type = context.get_constant(
        types.uint64, usm_ty_map[usm_ty] if usm_ty in usm_ty_map else 0
    )

    args = (
        context.get_dummy_value(),
        allocsize,
        usm_type,
        queue_ref_copy,
    )
    mip = types.MemInfoPointer(types.voidptr)
    arytypeclass = types.TypeRef(type(arrtype))
    sig = signature(
        mip,
        arytypeclass,
        types.intp,
        types.uint64,
        types.voidptr,
    )
    from numba_dpex.core.decorators import dpjit

    op = dpjit(_call_usm_allocator)
    fnop = context.typing_context.resolve_value_type(op)
    # The _call_usm_allocator function will be compiled and added to
    # registry when the get_call_type function is invoked.
    fnop.get_call_type(context.typing_context, sig.args, {})
    eqfn = context.get_function(fnop, sig)
    meminfo = eqfn(builder, args)
    data = context.nrt.meminfo_data(builder, meminfo)

    intp_t = context.get_value_type(types.intp)
    shape_array = cgutils.pack_array(builder, shapes, ty=intp_t)
    strides_array = cgutils.pack_array(builder, strides, ty=intp_t)

    _update_queue_attr(ary, queue=queue_ref_copy)
    populate_array(
        ary,
        data=builder.bitcast(data, datatype.as_pointer()),
        shape=shape_array,
        strides=strides_array,
        itemsize=itemsize,
        meminfo=meminfo,
    )

    return ary


@overload_classmethod(DpnpNdArray, "_usm_allocate", target=DPEX_TARGET_NAME)
def _ol_array_allocate(cls, allocsize, usm_type, queue):
    """Implements an allocator for dpnp.ndarrays."""

    def impl(cls, allocsize, usm_type, queue):
        return intrin_usm_alloc(allocsize, usm_type, queue)

    return impl


numba_config.DISABLE_PERFORMANCE_WARNINGS = 0


def _call_usm_allocator(arrtype, size, usm_type, queue):
    """Trampoline to call the intrinsic used for allocation"""
    return arrtype._usm_allocate(size, usm_type, queue)


numba_config.DISABLE_PERFORMANCE_WARNINGS = 1


@intrinsic(target=DPEX_TARGET_NAME)
def intrin_usm_alloc(typingctx, allocsize, usm_type, queue):
    """Intrinsic to call into the allocator for Array"""

    def codegen(context, builder, signature, args):
        [allocsize, usm_type, queue] = args
        dpexrtCtx = dpexrt.DpexRTContext(context)
        meminfo = dpexrtCtx.meminfo_alloc(builder, allocsize, usm_type, queue)
        return meminfo

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, usm_type, queue)
    return sig, codegen


def alloc_empty_arrayobj(context, builder, sig, queue_ref, args, is_like=False):
    """Construct an empty numba.np.arrayobj.make_array.<locals>.ArrayStruct

    Args:
        context (numba.core.base.BaseContext): One of the class derived
            from numba's BaseContext, e.g. CPUContext
        builder (llvmlite.ir.builder.IRBuilder): IR builder object from
            llvmlite.
        sig (numba.core.typing.templates.Signature): A numba's function
            signature object.
        queue_ref (llvmlite.ir.PointerType): Pointer to a DpctlSyclQueueRef
            object cast to i8*
        args (tuple): A tuple of args to be parsed as the arguments of
            an np.empty(), np.zeros() or np.ones() call.
        is_like (bool, optional): Decides on how to parse the args.
            Defaults to False.

    Returns: The LLVM IR value that stores the empty array
    """

    arrtype, shape = (
        _parse_empty_like_args(context, builder, sig, args)
        if is_like
        else _parse_empty_args(context, builder, sig, args)
    )
    ary = _empty_nd_impl(context, builder, arrtype, shape, queue_ref)

    return ary


def fill_arrayobj(context, builder, ary, arrtype, queue_ref, fill_value):
    """Fill a numba.np.arrayobj.make_array.<locals>.ArrayStruct
        with a specified value.

    Args:
        context (numba.core.base.BaseContext): One of the class derived
            from numba's BaseContext, e.g. CPUContext
        builder (llvmlite.ir.builder.IRBuilder): IR builder object from
            llvmlite.
        ary (numba.np.arrayobj.make_array.<locals>.ArrayStruct): A numba
            arrystruct allocated by numba's `make_array()` function.
        arrtype (tuple): Parsed arguments by numba's `_parse_empty_args`
            like functions for different numpy/dpnp methods, e.g. `zeros()`,
            `ones()`, `empty()`, and their corresponding `_like()` methods.
        fill_value (llvmlite.ir.values.Argument): An LLVMLite IR `Argument`
            object that specifies the values to be filled in.

    Returns:
        tuple(numba.np.arrayobj.make_array.<locals>.ArrayStruct,
            numba_dpex.core.types.dpnp_ndarray_type.DpnpNdArray):
                A tuple of allocated array and constructed array type info
                in DpnpNdArray.
    """

    itemsize = context.get_constant(types.intp, get_itemsize(context, arrtype))

    if isinstance(fill_value.type, DoubleType) or isinstance(
        fill_value.type, FloatType
    ):
        value_is_float = context.get_constant(types.boolean, 1)
    else:
        value_is_float = context.get_constant(types.boolean, 0)

    if isinstance(arrtype.dtype, types.scalars.Float):
        dest_is_float = context.get_constant(types.boolean, 1)
    else:
        dest_is_float = context.get_constant(types.boolean, 0)

    # Do a bitcast of the input to a 64-bit int.
    value = builder.bitcast(fill_value, llvmir.IntType(64))

    dpexrtCtx = dpexrt.DpexRTContext(context)
    dpexrtCtx.meminfo_fill(
        builder,
        ary.meminfo,
        itemsize,
        dest_is_float,
        value_is_float,
        value,
        queue_ref,
    )
    return ary, arrtype


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_empty(
    ty_context,
    ty_shape,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.empty().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_shape,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    sycl_queue_arg_pos = -2

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
        )

        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args
        )

        return ary._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_zeros(
    ty_context,
    ty_shape,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.zeros().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_shape,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )
    sycl_queue_arg_pos = -2

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
        )
        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args
        )
        fill_value = context.get_constant(types.intp, 0)
        ary, _ = fill_arrayobj(
            context,
            builder,
            ary,
            sig.return_type,
            qref_payload.queue_ref,
            fill_value,
        )

        return ary._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_ones(
    ty_context,
    ty_shape,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.ones().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_shape,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    sycl_queue_arg_pos = -2

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
        )
        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args
        )
        fill_value = context.get_constant(types.intp, 1)
        ary, _ = fill_arrayobj(
            context,
            builder,
            ary,
            sig.return_type,
            qref_payload.queue_ref,
            fill_value,
        )

        return ary._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_full(
    ty_context,
    ty_shape,
    ty_fill_value,
    ty_dtype,
    ty_order,
    ty_like,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.full().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_fill_value (numba.core.types.scalars): One of the Numba scalar
            types.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_like (numba.core.types.npytypes.Array): Numba type for array.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    signature = ty_retty(
        ty_shape,
        ty_fill_value,
        ty_dtype,
        ty_order,
        ty_like,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )
    sycl_queue_arg_pos = -2

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
        )
        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args
        )
        fill_value = context.get_argument_value(builder, sig.args[1], args[1])
        ary, _ = fill_arrayobj(
            context,
            builder,
            ary,
            sig.return_type,
            qref_payload.queue_ref,
            fill_value,
        )

        return ary._getvalue()

    return signature, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_empty_like(
    ty_context,
    ty_x1,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.empty_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_x1,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )
    sycl_queue_arg_pos = -2
    array_arg_pos = 0

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        array_arg = _ArgTyAndValue(sig.args[array_arg_pos], args[array_arg_pos])
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
            array_arg=array_arg,
        )

        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args, is_like=True
        )

        return ary._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_zeros_like(
    ty_context,
    ty_x1,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.zeros_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_x1,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    sycl_queue_arg_pos = -2
    array_arg_pos = 0

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        array_arg = _ArgTyAndValue(sig.args[array_arg_pos], args[array_arg_pos])
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
            array_arg=array_arg,
        )
        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args, is_like=True
        )
        fill_value = context.get_constant(types.intp, 0)
        ary, _ = fill_arrayobj(
            context,
            builder,
            ary,
            sig.return_type,
            qref_payload.queue_ref,
            fill_value,
        )

        return ary._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_ones_like(
    ty_context,
    ty_x1,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.ones_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_x1,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )
    sycl_queue_arg_pos = -2
    array_arg_pos = 0

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        array_arg = _ArgTyAndValue(sig.args[array_arg_pos], args[array_arg_pos])
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
            array_arg=array_arg,
        )
        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args, is_like=True
        )
        fill_value = context.get_constant(types.intp, 1)
        ary, _ = fill_arrayobj(
            context,
            builder,
            ary,
            sig.return_type,
            qref_payload.queue_ref,
            fill_value,
        )

        return ary._getvalue()

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def impl_dpnp_full_like(
    ty_context,
    ty_x1,
    ty_fill_value,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.full_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_fill_value (numba.core.types.scalars): One of the Numba scalar
            types.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    signature = ty_retty(
        ty_x1,
        ty_fill_value,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )
    sycl_queue_arg_pos = -2
    array_arg_pos = 0

    def codegen(context, builder, sig, args):
        sycl_queue_arg = _ArgTyAndValue(
            sig.args[sycl_queue_arg_pos], args[sycl_queue_arg_pos]
        )
        array_arg = _ArgTyAndValue(sig.args[array_arg_pos], args[array_arg_pos])
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
            array_arg=array_arg,
        )
        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args, is_like=True
        )
        fill_value = context.get_argument_value(builder, sig.args[1], args[1])
        ary, _ = fill_arrayobj(
            context,
            builder,
            ary,
            sig.return_type,
            qref_payload.queue_ref,
            fill_value,
        )

        return ary._getvalue()

    return signature, codegen
