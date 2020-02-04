// Maintainer: adulaney
#include "ExecutionConfiguration.h"
#include "HelperMath.h"
#include "MyGPUArray.h"
#include "VectorMath.h"

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __STATIC_STRUCTURE_FACTOR_GPU_H__
#define __STATIC_STRUCTURE_FACTOR_GPU_H__

/*! \file StaticStructureFactorGPU.h
    \brief Declares a class for computing static structure factor on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/attr.h>
#include <pybind11/pybind11.h>

class PYBIND11_EXPORT StaticStructureFactorGPU //: public StaticStructureFactor
{
public:
    //! Constructs the structure factor calculation
    StaticStructureFactorGPU(pybind11::list pos_lst,
                             pybind11::list k_lst,
                             int gpu_id);

    ~StaticStructureFactorGPU(){};

    pybind11::list compute();

protected:
    GPUArray<Scalar3> m_pos_vec; //! position vectors for each particle
    GPUArray<Scalar3> m_k_vec;   //! all wavevectors to probe
    int N;
    int N_k;

    GPUArray<Scalar> m_partial_sum_cos; //! Cos partial sum array
    GPUArray<Scalar> m_partial_sum_sin; //! Sin partial sum array

    GPUArray<Scalar> m_sum_cos; //! Cos sum array
    GPUArray<Scalar> m_sum_sin; //! Sin sum array

    unsigned int m_block_size;
    unsigned int m_num_blocks_k;
    unsigned int m_num_blocks_n;
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
};

void export_StaticStructureFactorGPU(pybind11::module &m);

#endif