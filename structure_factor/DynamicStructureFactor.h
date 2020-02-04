// Maintainer: adulaney

/*! \file DynamicStructureFactor.h
    \brief Declares a class for computing dynamic structure factor on the CPU.
*/

#include "ExecutionConfiguration.h"
#include "HelperMath.h"
#include "MyGPUArray.h"
#include "VectorMath.h"

#ifndef __DYNAMIC_STRUCTURE_FACTOR_H__
#define __DYNAMIC_STRUCTURE_FACTOR_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

class PYBIND11_EXPORT DynamicStructureFactor
{
public:
    //! Constructs the structure factor calculation
    DynamicStructureFactor(pybind11::list init_pos_lst,
                           pybind11::list pos_lst,
                           pybind11::list k_lst);

    ~DynamicStructureFactor();

    pybind11::list compute();

protected:
    GPUArray<Scalar3> m_init_pos_vec; //! initial position vectors for each particle
    GPUArray<Scalar3> m_pos_vec;      //! position vectors for each particle
    GPUArray<Scalar3> m_k_vec;        //! all wavevectors to probe
    int N;
    int N_k;
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
};

void export_DynamicStructureFactor(pybind11::module &m);

#endif