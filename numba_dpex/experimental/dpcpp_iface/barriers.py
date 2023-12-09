import numba
from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

from numba_dpex.core import itanium_mangler as ext_itanium_mangler

from ._spv_atomic_helper import get_memory_semantics_mask, get_scope
from .memory_enums import MemoryOrder, MemoryScope


def group_barrier(fence_scope=MemoryScope.work_group.value):
    """The function for performing barrier operation
    across all work-items in a work group.
    Modeled after ``sycl::group_barrier`` function
    that take a work group as argument.

    Args:
    fence_scope (optional): scope of any memory consistency
    operations that are performed by the barrier.

    """
    pass


def sub_group_barrier(fence_scope=MemoryScope.sub_group):
    """The function for performing barrier operation
    across all work-items in a sub-group.
    Modeled after ``sycl::group_barrier`` function
    that takes a sub-group as argument.

    Args:
    fence_scope (optional): scope of any memory consistency
    operations that are performed by the barrier.

    """

    pass


@intrinsic
def _intrinsic_barrier(
    ty_context, ty_exec_scope, ty_mem_scope, ty_spirv_mem_sem_mask
):
    sig = types.void(ty_exec_scope, ty_mem_scope, ty_spirv_mem_sem_mask)

    def codegen(context, builder, sig, args):
        fn_name = "__spirv_ControlBarrier"
        mangled_fn_name = ext_itanium_mangler.mangle_ext(
            fn_name, [ty_exec_scope, ty_mem_scope, ty_spirv_mem_sem_mask]
        )

        spirv_fn_arg_types = [context.get_value_type(t) for t in sig.args]

        fnty = llvmir.FunctionType(llvmir.VoidType(), spirv_fn_arg_types)

        fn_args = [args[0], args[1], args[2]]

        fn = cgutils.get_or_insert_function(
            builder.module, fnty, mangled_fn_name
        )
        # XXX Uncomment once the llvmlite PR#1019 is merged and available for use
        # fn.attributes.add("convergent")
        # fn.attributes.add("nounwind")
        fn.calling_convention = "spir_func"

        builder.call(fn, fn_args)

        # XXX Uncomment once the llvmlite PR#1019 is merged and available for use
        # callinst.attributes.add("convergent")
        # callinst.attributes.add("nounwind")

        return

    return (
        sig,
        codegen,
    )


def get_memory_scope(fence_scope):
    if isinstance(fence_scope, numba.types.Literal):
        return get_scope(fence_scope.literal_value)
    else:
        return get_scope(fence_scope.value)


@overload(group_barrier, prefer_literal=True, inline="always")
def ol_group_barrier(fence_scope=MemoryScope.work_group):
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        MemoryOrder.seq_cst.value
    )
    exec_scope = get_scope(MemoryScope.work_group.value)
    mem_scope = get_memory_scope(fence_scope)

    def impl(fence_scope=MemoryScope.work_group):
        return _intrinsic_barrier(
            exec_scope, mem_scope, spirv_memory_semantics_mask
        )

    return impl


@overload(sub_group_barrier, prefer_literal=True, inline="always")
def ol_sub_group_barrier(fence_scope=MemoryScope.sub_group):
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        MemoryOrder.seq_cst.value
    )
    exec_scope = get_scope(MemoryScope.sub_group.value)
    mem_scope = get_memory_scope(fence_scope)

    def impl(fence_scope=MemoryScope.sub_group):
        return _intrinsic_barrier(
            exec_scope, mem_scope, spirv_memory_semantics_mask
        )

    return impl
