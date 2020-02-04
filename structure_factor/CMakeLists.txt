# Maintainer: adulaney

########################################################################
# Library setup
########################################################################
option(ENABLE_CUDA "Enable the compilation of CUDA GPU code" OFF)

set(COMPONENT_NAME structure_factor)

set(PYBIND11_CPP_STANDARD -std=c++14)

set(_${COMPONENT_NAME}_sources
    module.cc
    StaticStructureFactor.cc
    DynamicStructureFactor.cc
    ExecutionConfiguration.cc
    )

    set(_${COMPONENT_NAME}_headers
    StaticStructureFactor.h 
    DynamicStructureFactor.h 
    MyGPUArray.h 
    HelperMath.h 
    VectorMath.h 
    CachedAllocator.h 
    ExecutionConfiguration.h 
    )
    
if (CMAKE_VERSION VERSION_GREATER 3.0.99)
    # let cmake set the flags
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    # findcuda needs to be told explicitly to use c++11 as CMAKE_CXX_STANDARD does not modify CMAKE_CXX_FLAGS
    list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
else()
    # manually enable flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()

if (ENABLE_CUDA)
    add_definitions(-DENABLE_CUDA)
endif()

if (ENABLE_CUDA)
    enable_language(CUDA)
    FIND_PACKAGE(CUDA 7.0 REQUIRED)
    include_directories(${CUDA_INCLUDE})
    include_directories(${CUDA_INCLUDE_DIRS})
    get_directory_property(DIRS INCLUDE_DIRECTORIES SYSTEM)

    mark_as_advanced(CUDA_SDK_ROOT_DIR)
    if (CUDA_TOOLKIT_ROOT_DIR)
        mark_as_advanced(CUDA_TOOLKIT_ROOT_DIR)
    endif (CUDA_TOOLKIT_ROOT_DIR)
    mark_as_advanced(CUDA_VERBOSE_BUILD)
    mark_as_advanced(CUDA_BUILD_EMULATION)
    mark_as_advanced(CUDA_HOST_COMPILER)
    mark_as_advanced(CUDA_dl_LIBRARY)
    mark_as_advanced(CUDA_rt_LIBRARY)

    mark_as_advanced(THRUST_INCLUDE_DIR)

    if (ENABLE_NVTOOLS)
        find_library(CUDA_nvToolsExt_LIBRARY
                     NAMES nvToolsExt
                     PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
                           "${CUDA_TOOLKIT_ROOT_DIR}/lib"
                     ENV CUDA_LIB_PATH
                     DOC "nvTools library"
                     NO_DEFAULT_PATH
                     )

        mark_as_advanced(CUDA_nvToolsExt_LIBRARY)
    endif(ENABLE_NVTOOLS)
endif(ENABLE_CUDA)

if (ENABLE_CUDA)
    # setup nvcc to build for all CUDA architectures. Allow user to modify the list if desired
    if (CUDA_VERSION VERSION_GREATER 8.99)
        set(CUDA_ARCH_LIST 30 35 50 60 70 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    elseif (CUDA_VERSION VERSION_GREATER 7.99)
        set(CUDA_ARCH_LIST 30 35 50 60 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    elseif (CUDA_VERSION VERSION_GREATER 6.99)
        set(CUDA_ARCH_LIST 30 35 50 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    endif()

    foreach(_cuda_arch ${CUDA_ARCH_LIST})
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_arch},code=sm_${_cuda_arch}")
    endforeach (_cuda_arch)

    # need to know the minumum supported CUDA_ARCH
    set(_cuda_arch_list_sorted ${CUDA_ARCH_LIST})
    list(SORT _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_min_arch)
    list(GET _cuda_arch_list_sorted -1 _cuda_max_arch)
    add_definitions(-DCUDA_ARCH=${_cuda_min_arch})

    if (_cuda_min_arch LESS 30)
        message(SEND_ERROR "Package requires compute 3.0 or newer")
    endif ()

    # only generage ptx code for the maximum supported CUDA_ARCH (saves on file size)
    list(REVERSE _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_max_arch)
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_max_arch},code=compute_${_cuda_max_arch}")

endif (ENABLE_CUDA)

if (ENABLE_CUDA)
    list(APPEND _${COMPONENT_NAME}_sources DynamicStructureFactorGPU.cc
                                           StaticStructureFactorGPU.cc
        )
    list(APPEND _${COMPONENT_NAME}_headers DynamicStructureFactorGPU.cuh
                                           DynamicStructureFactorGPU.h
                                           StaticStructureFactorGPU.cuh
                                           StaticStructureFactorGPU.h
        )

    set(_${COMPONENT_NAME}_cu_sources DynamicStructureFactorGPU.cu
                                      StaticStructureFactorGPU.cu
        )
endif (ENABLE_CUDA)

if (ENABLE_CUDA)
    CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${COMPONENT_NAME}_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module (_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} ${PYTHON_LIBRARIES})

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
    )
ENDMACRO(copy_file)

set(files
    __init__.py
    structure_class.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})
