from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

from numba_dpex.core import itanium_mangler as ext_itanium_mangler
from numba_dpex.core.targets.kernel_target import (
    CC_SPIR_FUNC,
    DPEX_KERNEL_TARGET_NAME,
)

from ._spv_atomic_helper import get_memory_semantics_mask, get_scope


def atomic_fence(memory_order, memory_scope):
    """The function for performing memory fence across all work-items.
    Modeled after ``sycl::atomic_fence`` function.

    Args:
    memory_order: The memory synchronization order.

    memory_scope: The set of work-items and devices to which
    the memory ordering constraints of a given atomic operation apply.

    """
    pass


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_atomic_fence(ty_context, ty_spirv_mem_sem_mask, ty_spirv_scope):
    sig = types.void(ty_spirv_scope, ty_spirv_mem_sem_mask)

    def codegen(context, builder, sig, args):
        name = "__spirv_MemoryBarrier"
        mangled_fn_name = ext_itanium_mangler.mangle_ext(
            name, [ty_spirv_scope, ty_spirv_mem_sem_mask]
        )

        spirv_fn_arg_types = [context.get_value_type(t) for t in sig.args]

        fnty = llvmir.FunctionType(llvmir.VoidType(), spirv_fn_arg_types)

        fn_args = [
            args[1],
            args[0],
        ]

        fn = cgutils.get_or_insert_function(
            builder.module, fnty, mangled_fn_name
        )
        fn.calling_convention = CC_SPIR_FUNC

        builder.call(fn, fn_args)

    return (
        sig,
        codegen,
    )


@overload(
    atomic_fence,
    prefer_literal=True,
    inline="always",
    target=DPEX_KERNEL_TARGET_NAME,
)
def ol_atomic_fence(memory_order, memory_scope):
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        memory_order.literal_value
    )
    spirv_scope = get_scope(memory_scope.literal_value)

    def ol_atomic_fence_impl(memory_order, memory_scope):
        return _intrinsic_atomic_fence(spirv_memory_semantics_mask, spirv_scope)

    return ol_atomic_fence_impl
