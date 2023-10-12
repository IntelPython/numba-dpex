#include <Python.h>
#include <numpy/npy_common.h>
#include <numba/_arraystruct.h>

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void NUMBA_DPEX_SYCL_KERNEL_init_sequence_step_dispatch_vectors();
    void NUMBA_DPEX_SYCL_KERNEL_init_affine_sequence_dispatch_vectors();
    uint NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence(
        void *start,
        void *dt,
        arystruct_t *dst,
        int ndim,
        u_int8_t is_c_contiguous,
        const DPCTLSyclQueueRef exec_q);
    // const DPCTLEventVectorRef depends = std::vector<DPCTLSyclEventRef>());

#ifdef __cplusplus
}
#endif
