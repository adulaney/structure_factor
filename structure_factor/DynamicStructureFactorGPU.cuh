// Maintainer: adulaney

#include "HelperMath.h"
#include <cuda_runtime.h>

#ifndef __DYNAMIC_STRUCTURE_FACTOR_GPU_CUH__
#define __DYNAMIC_STRUCTURE_FACTOR_GPU_CUH__

cudaError_t gpu_dynamic_structure_factor_compute(const Scalar3 *d_init_pos,
                                                 const Scalar3 *d_pos,
                                                 const Scalar3 *d_k,
                                                 Scalar *d_partial_sum_cos,
                                                 Scalar *d_partial_sum_sin,
                                                 Scalar *d_partial_sum_init_cos,
                                                 Scalar *d_partial_sum_init_sin,
                                                 Scalar *d_sum_cos,
                                                 Scalar *d_sum_sin,
                                                 Scalar *d_sum_init_cos,
                                                 Scalar *d_sum_init_sin,
                                                 unsigned int num_blocks_n,
                                                 unsigned int num_blocks_k,
                                                 int N,
                                                 int N_k);

#endif //__DYNAMIC_STRUCTURE_FACTOR_GPU_CUH__
