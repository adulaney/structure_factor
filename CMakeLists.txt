# Maintainer: adulaney
# IF(ENABLE_CUDA)
PROJECT(structure_factor LANGUAGES CXX)
# ELSE()
# PROJECT(structure_factor LANGUAGES CXX)
# ENDIF()

CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )

LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/Modules
  
)

########################################################################
# Macros and additional flags
########################################################################

INCLUDE( CheckCXXCompilerFlag )

FILE( GLOB filenames "cmake/Macros/*.cmake" )
FOREACH( filename ${filenames} )
  INCLUDE( ${filename} )
ENDFOREACH()

FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs   3)
FIND_PACKAGE(pybind11 REQUIRED PATHS PythonInterp)

IF( PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND PYBIND11_FOUND )
  INCLUDE_DIRECTORIES(
    ${PYTHON_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIRS}
  )
ENDIF()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_INSTALL_RPATH "$ORIGIN/..:$ORIGIN")

set(PYTHON_MODULE_BASE_DIR ${CMAKE_INSTALL_PREFIX})
message(STATUS "Install plugin to: " ${PYTHON_MODULE_BASE_DIR})

### add subdirectories
add_subdirectory(${PROJECT_NAME})
