// Maintainer: adulaney

// #include

/*! \file StaticStructureFactor.h
    \brief Declares a class for computing static structure factor on the CPU.
*/

#include "ExecutionConfiguration.h"
#include "HelperMath.h"
#include "MyGPUArray.h"
#include "VectorMath.h"

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __STATIC_STRUCTUREFACTOR_H__
#define __STATIC_STRUCTUREFACTOR_H__

class PYBIND11_EXPORT StaticStructureFactor
{
public:
    //! Constructs the structure factor calculation
    StaticStructureFactor(pybind11::list pos_lst,
                          pybind11::list k_lst);

    ~StaticStructureFactor();

    pybind11::list compute();

protected:
    GPUArray<Scalar3> m_pos_vec; //! position vectors for each particle
    GPUArray<Scalar3> m_k_vec;   //! all wavevectors to probe
    Scalar N;
    Scalar N_k;
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
};

void export_StaticStructureFactor(pybind11::module &m);

#endif