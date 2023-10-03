#include "sequences.hpp"
#include "dispatch.hpp"
#include "types.hpp"

static ndpx::runtime::kernel::tensor::sequence_step_opaque_ptr_t
    sequence_step_dispatch_vector[ndpx::runtime::kernel::types::num_types];

void init_sequence_dispatch_vectors(void)
{

    ndpx::runtime::kernel::dispatch::DispatchVectorBuilder<
        ndpx::runtime::kernel::tensor::sequence_step_opaque_ptr_t,
        ndpx::runtime::kernel::tensor::SequenceStepFactory,
        ndpx::runtime::kernel::types::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(sequence_step_dispatch_vector);
}
