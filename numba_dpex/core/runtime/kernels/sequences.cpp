#include "sequences.hpp"
#include "api.h"
#include "dispatch.hpp"
#include "types.hpp"
#include <iostream>

static ndpx::runtime::kernel::tensor::sequence_step_ptr_t
    sequence_step_dispatch_vector[ndpx::runtime::kernel::types::num_types];

static ndpx::runtime::kernel::tensor::affine_sequence_step_ptr_t
    affine_sequence_step_dispatch_vector
        [ndpx::runtime::kernel::types::num_types];

void init_sequence_dispatch_vectors(void)
{
    ndpx::runtime::kernel::dispatch::DispatchVectorBuilder<
        ndpx::runtime::kernel::tensor::sequence_step_ptr_t,
        ndpx::runtime::kernel::tensor::SequenceStepFactory,
        ndpx::runtime::kernel::types::num_types>
        dvb;
    dvb.populate_dispatch_vector(sequence_step_dispatch_vector);
    std::cout << "-----> init_sequence_dispatch_vectors()" << std::endl;
}

void init_affine_sequence_dispatch_vectors(void)
{
    ndpx::runtime::kernel::dispatch::DispatchVectorBuilder<
        ndpx::runtime::kernel::tensor::affine_sequence_step_ptr_t,
        ndpx::runtime::kernel::tensor::AffineSequenceStepFactory,
        ndpx::runtime::kernel::types::num_types>
        dvb;
    dvb.populate_dispatch_vector(affine_sequence_step_dispatch_vector);
    std::cout << "-----> init_affine_sequence_dispatch_vectors()" << std::endl;
}
