// Maintainer: adulaney

#include "StaticStructureFactor.h"
#include "HelperMath.h"
#include <vector>

// #include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

/*! \file StaticStructureFactor.cc
    \brief Declares GPU kernel code for computing static structure factor on the GPU. 
*/

//! Kernel for computing static structure factor on the GPU
/*! \param h_pos An array of (x,y,z) tuples for particle positions
    \param k_lst An array of (x,y,z) tuples for the possible wavevectors 

*/

StaticStructureFactor::StaticStructureFactor(pybind11::list pos_lst,
                                             pybind11::list k_lst)
    : N(0), N_k(0)
{
    // m_exec_conf = std::shared_ptr<const ExecutionConfiguration>(new ExecutionConfiguration());
    m_exec_conf = std::shared_ptr<const ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::executionMode::CPU, -1, false, false, (unsigned int)0));

    vector<Scalar3> c_pos_lst;
    py::tuple tmp_pos;
    for (unsigned int i = 0; i < len(pos_lst); i++)
    {
        tmp_pos = py::cast<py::tuple>(pos_lst[i]);
        if (len(tmp_pos) != 3)
            throw runtime_error("Non-3D position given for StaticStructureFactor");
        c_pos_lst.push_back(make_scalar3(py::cast<Scalar>(tmp_pos[0]), py::cast<Scalar>(tmp_pos[1]), py::cast<Scalar>(tmp_pos[2])));
    }

    vector<Scalar3> c_k_lst;
    py::tuple tmp_k;

    for (unsigned int i = 0; i < len(k_lst); i++)
    {
        tmp_k = py::cast<py::tuple>(k_lst[i]);
        if (len(tmp_k) != 3)
            throw runtime_error("Non-3D wavevector given for StaticStructureFactor");
        c_k_lst.push_back(make_scalar3(py::cast<Scalar>(tmp_k[0]), py::cast<Scalar>(tmp_k[1]), py::cast<Scalar>(tmp_k[2])));
    }

    N = c_pos_lst.size();
    N_k = c_k_lst.size();

    GPUArray<Scalar3> tmp_posvec(len(pos_lst), m_exec_conf);
    GPUArray<Scalar3> tmp_kvec(len(k_lst), m_exec_conf);

    m_pos_vec.swap(tmp_posvec);
    m_k_vec.swap(tmp_kvec);

    ArrayHandle<Scalar3> h_pos_vec(m_pos_vec, access_location::host);
    ArrayHandle<Scalar3> h_k_vec(m_k_vec, access_location::host);

    for (unsigned int i = 0; i < len(pos_lst); i++)
    {
        h_pos_vec.data[i] = make_scalar3(0, 0, 0);
        h_pos_vec.data[i].x = c_pos_lst[i].x;
        h_pos_vec.data[i].y = c_pos_lst[i].y;
        h_pos_vec.data[i].z = c_pos_lst[i].z;
    }

    for (unsigned int i = 0; i < len(k_lst); i++)
    {
        h_k_vec.data[i] = make_scalar3(0, 0, 0);
        h_k_vec.data[i].x = c_k_lst[i].x;
        h_k_vec.data[i].y = c_k_lst[i].y;
        h_k_vec.data[i].z = c_k_lst[i].z;
    }
}

StaticStructureFactor::~StaticStructureFactor()
{
}

py::list StaticStructureFactor::compute()
{
    //array handles
    ArrayHandle<Scalar3> h_pos_vec(m_pos_vec, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k_vec(m_k_vec, access_location::host, access_mode::read);

    py::list ssf;

    for (unsigned int i = 0; i < N_k; ++i)
    {
        Scalar cos_n = 0.0;
        Scalar sin_n = 0.0;
        for (unsigned int j = 0; j < N; ++j)
        {
            Scalar3 pos;
            pos = make_scalar3(h_pos_vec.data[j].x, h_pos_vec.data[j].y, h_pos_vec.data[j].z);

            Scalar3 kvec;
            kvec = make_scalar3(h_k_vec.data[i].x, h_k_vec.data[i].y, h_k_vec.data[i].z);

            Scalar phase;
            phase = kvec.x * pos.x + kvec.y * pos.y + kvec.z * pos.z;

            cos_n += slow::cos(phase);
            sin_n += slow::sin(phase);
        }
        ssf.append(py::float_((cos_n * cos_n + sin_n * sin_n) / float(N)));
    }
    return ssf;
}

void export_StaticStructureFactor(py::module &m)
{
    py::class_<StaticStructureFactor, std::shared_ptr<StaticStructureFactor>>(m, "StaticStructureFactor")
        .def(py::init<py::list, py::list>())
        .def("compute", &StaticStructureFactor::compute);
}