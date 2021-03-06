// Maintainer: adulaney


#include "StaticStructureFactorGPU.cuh"

#include <assert.h>

/*! \file StaticStructureFactorGPU.cu
    \brief Declares GPU kernel code for computing static structure factor on the GPU. 
*/

//! Kernel for computing static structure factor on the GPU
/*! \param 
*/

extern "C" __global__ void gpu_static_structure_factor_compute_kernel(const Scalar3 *d_pos,
                                                                    const Scalar3 *d_k,
                                                                    Scalar *d_partial_sum_cos,
                                                                    Scalar *d_partial_sum_sin,
                                                                    int N,
                                                                    int N_k)
{
    __shared__ Scalar cos_sdata[32][32];
    __shared__ Scalar sin_sdata[32][32];
    
    int N_id = threadIdx.y + blockIdx.y * blockDim.y;
    int k_id = threadIdx.x + blockIdx.x * blockDim.x;
    if ((N_id < N) && (k_id < N_k)){
            
        Scalar3 pos = d_pos[N_id];

        Scalar3 kvec = d_k[k_id];
        Scalar phase = kvec.x * pos.x + kvec.y * pos.y + kvec.z * pos.z;
        
        __syncthreads();
        cos_sdata[threadIdx.x][threadIdx.y] = slow::cos(phase);
        sin_sdata[threadIdx.x][threadIdx.y] = slow::sin(phase);
        __syncthreads();

        // Reduce the sum in parallel
        int offs = blockDim.y >> 1;
        while (offs > 0){
            if (threadIdx.y < offs){
                cos_sdata[threadIdx.x][threadIdx.y] += cos_sdata[threadIdx.x][threadIdx.y + offs];
                sin_sdata[threadIdx.x][threadIdx.y] += sin_sdata[threadIdx.x][threadIdx.y + offs];
            }
            offs >>= 1;
            __syncthreads();
        }
        if (threadIdx.y == 0){
            d_partial_sum_cos[blockIdx.y * N_k + k_id] = cos_sdata[threadIdx.x][0];
            d_partial_sum_sin[blockIdx.y * N_k + k_id] = sin_sdata[threadIdx.x][0];
        }

    }

}

extern "C" __global__ void gpu_reduce_static_partial_sum_kernel(Scalar *d_partial_sum_cos,
                                                        Scalar *d_partial_sum_sin,
                                                        Scalar *d_sum_cos,
                                                        Scalar *d_sum_sin,
                                                        unsigned int num_blocks_n,
                                                        int N_k)
{
    __shared__ Scalar cos_sdata[32][32];
    __shared__ Scalar sin_sdata[32][32];

    int k_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (k_id < N_k){
        Scalar sum_cos = Scalar(0.0);
        Scalar sum_sin = Scalar(0.0);
        
        for (int start = 0; start < num_blocks_n; start += blockDim.y){
            __syncthreads();
            if (start + threadIdx.y < num_blocks_n){
                cos_sdata[threadIdx.x][threadIdx.y] = d_partial_sum_cos[k_id + N_k*(start + threadIdx.y)];
                sin_sdata[threadIdx.x][threadIdx.y] = d_partial_sum_sin[k_id + N_k*(start + threadIdx.y)];
            }
            else{
                cos_sdata[threadIdx.x][threadIdx.y] = Scalar(0.0);
                sin_sdata[threadIdx.x][threadIdx.y] = Scalar(0.0);
            }
            __syncthreads();

            // reduce sum in parallel
            int offs = blockDim.y >> 1;
            while (offs > 0){
                if (threadIdx.y < offs){
                    cos_sdata[threadIdx.x][threadIdx.y] += cos_sdata[threadIdx.x][threadIdx.y + offs];
                    sin_sdata[threadIdx.x][threadIdx.y] += sin_sdata[threadIdx.x][threadIdx.y + offs];
                }
                offs >>= 1;
                __syncthreads();
            }

            sum_cos += cos_sdata[threadIdx.x][0];
            sum_sin += sin_sdata[threadIdx.x][0];
        }
        
        if (threadIdx.y == 0){
            d_sum_cos[k_id] = sum_cos;
            d_sum_sin[k_id] = sum_sin;
        }
    }
}


cudaError_t gpu_static_structure_factor_compute(const Scalar3 *d_pos,
                                                const Scalar3 *d_k,
                                                Scalar *d_partial_sum_cos,
                                                Scalar *d_partial_sum_sin,
                                                Scalar *d_sum_cos,
                                                Scalar *d_sum_sin,
                                                unsigned int num_blocks_n,
                                                unsigned int num_blocks_k,
                                                int N,
                                                int N_k)
{
    // define grid to run kernel
    int block_size = 32;

    dim3 blocks( num_blocks_k, num_blocks_n);
    dim3 threads(block_size, block_size);

    dim3 blocks1(num_blocks_k, 1);
    dim3 threads1(block_size, block_size);

    // run the kernel
    gpu_static_structure_factor_compute_kernel<<< blocks, threads>>>(d_pos,
                                                                        d_k,
                                                                        d_partial_sum_cos,
                                                                        d_partial_sum_sin,
                                                                        N,
                                                                        N_k);

    // run the summation kernel
    gpu_reduce_static_partial_sum_kernel<<< blocks1, threads1>>>(d_partial_sum_cos,
                                                        d_partial_sum_sin,
                                                        d_sum_cos,
                                                        d_sum_sin,
                                                        num_blocks_n,
                                                        N_k);
    
    return cudaSuccess;
}