// Maintainer: adulaney

#include "StaticStructureFactorGPU.h"
#include "StaticStructureFactorGPU.cuh"
#include <assert.h>
#include "ExecutionConfiguration.h"
#include <string>
#include <memory>
using namespace std;
namespace py = pybind11;

/*! \file StaticStructureFactorGPU.cc
    \brief Declares GPU kernel code for computing static structure factor on the GPU. 
*/

//! Kernel for computing static structure factor on the GPU
/*! \param h_pos An array of (x,y,z) tuples for particle positions
    \param k_lst An array of (x,y,z) tuples for the possible wavevectors 
    \param t_lst An array of times
    \param bin_size specifies the size of wavenumber bins to use

*/
StaticStructureFactorGPU::StaticStructureFactorGPU(pybind11::list h_pos_lst, pybind11::list k_lst, int gpu_id)
{
    // Define execution configuration
    m_exec_conf = std::shared_ptr<const ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::executionMode::GPU, gpu_id, false, false, (unsigned int)0));

    if (!m_exec_conf->isCUDAEnabled())
    {
        throw runtime_error("Error Initializing StaticStructureFactorGPU");
    }

    // Initialize gpu kernel parameters
    m_block_size = 32;
    m_num_blocks_n = len(h_pos_lst) / m_block_size + 1;
    m_num_blocks_k = len(k_lst) / m_block_size + 1;

    // Initialize sum arrays
    GPUArray<Scalar> tmp_partial_sum_cos(len(k_lst) * m_num_blocks_n, m_exec_conf);
    GPUArray<Scalar> tmp_partial_sum_sin(len(k_lst) * m_num_blocks_n, m_exec_conf);

    m_partial_sum_cos.swap(tmp_partial_sum_cos);
    m_partial_sum_sin.swap(tmp_partial_sum_sin);

    GPUArray<Scalar> tmp_sum_cos(len(k_lst), m_exec_conf);
    GPUArray<Scalar> tmp_sum_sin(len(k_lst), m_exec_conf);

    m_sum_cos.swap(tmp_sum_cos);
    m_sum_sin.swap(tmp_sum_sin);

    // Read in positions
    vector<Scalar3> c_pos_lst;

    py::tuple tmp_pos;

    for (unsigned int i = 0; i < len(h_pos_lst); i++)
    {
        tmp_pos = py::cast<py::tuple>(h_pos_lst[i]);

        if ((len(tmp_pos) != 3))
            throw runtime_error("Non-3D position given for StaticStructureFactor");

        c_pos_lst.push_back(make_scalar3(py::cast<Scalar>(tmp_pos[0]), py::cast<Scalar>(tmp_pos[1]), py::cast<Scalar>(tmp_pos[2])));
    }

    // Read in wavevectors
    vector<Scalar3> c_k_lst;
    py::tuple tmp_k;

    for (unsigned int i = 0; i < len(k_lst); i++)
    {
        tmp_k = py::cast<py::tuple>(k_lst[i]);
        if (len(tmp_k) != 3)
            throw runtime_error("Non-3D wavevector given for StaticStructureFactor");

        c_k_lst.push_back(make_scalar3(py::cast<Scalar>(tmp_k[0]), py::cast<Scalar>(tmp_k[1]), py::cast<Scalar>(tmp_k[2])));
    }

    // Determine size of arrays for gpu call
    N = c_pos_lst.size();
    N_k = c_k_lst.size();

    // Initialize GPU arrays for position and wavevectors
    GPUArray<Scalar3> tmp_posvec(len(h_pos_lst), m_exec_conf);
    GPUArray<Scalar3> tmp_kvec(len(k_lst), m_exec_conf);

    m_pos_vec.swap(tmp_posvec);
    m_k_vec.swap(tmp_kvec);

    ArrayHandle<Scalar3> h_pos_vec(m_pos_vec, access_location::host);
    ArrayHandle<Scalar3> h_k_vec(m_k_vec, access_location::host);

    // Write positions to GPUArrays
    for (unsigned int i = 0; i < len(h_pos_lst); i++)
    {
        h_pos_vec.data[i] = make_scalar3(c_pos_lst[i].x, c_pos_lst[i].y, c_pos_lst[i].z);
    }

    // Write wavevectors to GPUArrays
    for (unsigned int i = 0; i < len(k_lst); i++)
    {
        h_k_vec.data[i] = make_scalar3(c_k_lst[i].x, c_k_lst[i].y, c_k_lst[i].z);
    }
}

py::list StaticStructureFactorGPU::compute()
{
    {
        //array handles
        ArrayHandle<Scalar3> d_pos_vec(m_pos_vec, access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_k_vec(m_k_vec, access_location::device, access_mode::read);

        ArrayHandle<Scalar> d_partial_sum_cos(m_partial_sum_cos, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_partial_sum_sin(m_partial_sum_sin, access_location::device, access_mode::overwrite);

        ArrayHandle<Scalar> d_sum_cos(m_sum_cos, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sum_sin(m_sum_sin, access_location::device, access_mode::overwrite);

        // perform calculation on GPU
        gpu_static_structure_factor_compute(d_pos_vec.data,
                                            d_k_vec.data,
                                            d_partial_sum_cos.data,
                                            d_partial_sum_sin.data,
                                            d_sum_cos.data,
                                            d_sum_sin.data,
                                            m_num_blocks_n,
                                            m_num_blocks_k,
                                            N,
                                            N_k);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
            CHECK_CUDA_ERROR();
        }
    }
    {
        // Read in the parital sums
        ArrayHandle<Scalar> h_sum_cos(m_sum_cos, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_sum_sin(m_sum_sin, access_location::host, access_mode::read);

        py::list ssf;

        // Finish summing over particles
        for (int i = 0; i < N_k; i++)
        {
            Scalar sum_cos = 0.0;
            Scalar sum_sin = 0.0;

            sum_cos += h_sum_cos.data[i];
            sum_sin += h_sum_sin.data[i];

            ssf.append(py::float_((sum_cos * sum_cos + sum_sin * sum_sin) / float(N)));
        }
        return ssf;
    }
}

void export_StaticStructureFactorGPU(py::module &m)
{
    py::class_<StaticStructureFactorGPU, std::shared_ptr<StaticStructureFactorGPU>>(m, "StaticStructureFactorGPU")
        .def(py::init<py::list, py::list, int>(), py::return_value_policy::move)
        .def("compute", &StaticStructureFactorGPU::compute, py::return_value_policy::move);
}
