// Maintainer: adulaney

#include "HelperMath.h"
#include <cuda_runtime.h>

#ifndef __STATIC_STRUCTURE_FACTOR_GPU_CUH__
#define __STATIC_STRUCTURE_FACTOR_GPU_CUH__

cudaError_t gpu_static_structure_factor_compute(const Scalar3 *d_pos,
                                                const Scalar3 *d_k,
                                                Scalar *d_partial_sum_cos,
                                                Scalar *d_partial_sum_sin,
                                                Scalar *d_sum_cos,
                                                Scalar *d_sum_sin,
                                                unsigned int num_blocks_n,
                                                unsigned int num_blocks_k,
                                                int N,
                                                int N_k);

#endif //__STATIC_STRUCTURE_FACTOR_GPU_CUH__
