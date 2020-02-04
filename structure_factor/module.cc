// Include defined classes to be exported to python
#ifdef ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#include "HelperMath.h"
#include "ExecutionConfiguration.h"

#include "CachedAllocator.h"
#include "MyGPUArray.h"

#include "StaticStructureFactor.h"
#include "DynamicStructureFactor.h"

// #include "VectorMath.h"
// #include <vector>
// #include "cudacpu_host_defines.h"
// #include "cudacpu_vector_types.h"
// #include "cudacpu_vector_functions.h"

// Include GPU classes
#ifdef ENABLE_CUDA
#include "StaticStructureFactorGPU.h"
#include "DynamicStructureFactorGPU.h"
#endif

#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_structure_factor, m)
{
    export_ExecutionConfiguration(m);
    export_StaticStructureFactor(m);
    export_DynamicStructureFactor(m);

#ifdef ENABLE_CUDA
    export_StaticStructureFactorGPU(m);
    export_DynamicStructureFactorGPU(m);
#endif
}