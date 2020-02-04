// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file MyGPUArray.h
    \brief Defines the GPUArray class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __MYGPUARRAY_H__
#define __MYGPUARRAY_H__

// 4 GB is considered a large allocation for a single GPU buffer, and user should be warned
#define LARGEALLOCBYTES 0xffffffff

// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "ExecutionConfiguration.h"
#include <string.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <stdlib.h>
#include <assert.h>

//! Specifies where to acquire the data
struct access_location
{
    //! The enum
    enum Enum
    {
        host, //!< Ask to acquire the data on the host
#ifdef ENABLE_CUDA
        device //!< Ask to acquire the data on the device
#endif
    };
};

//! Defines where the data is currently stored
struct data_location
{
    //! The enum
    enum Enum
    {
        host, //!< Data was last updated on the host
#ifdef ENABLE_CUDA
        device,    //!< Data was last updated on the device
        hostdevice //!< Data is up to date on both the host and device
#endif
    };
};

//! Sepcify how the data is to be accessed
struct access_mode
{
    //! The enum
    enum Enum
    {
        read,      //!< Data will be accessed read only
        readwrite, //!< Data will be accessed for read and write
        overwrite  //!< The data is to be completely overwritten during this aquire
    };
};

template <class T>
class GPUArray;

//! Handle to access the data pointer handled by GPUArray
/*! The data in GPUArray is only accessible via ArrayHandle. The pointer is accessible for the lifetime of the
    ArrayHandle. When the ArrayHandle is destroyed, the GPUArray is notified that the data has been released. This
    tracking mechanism provides for error checking that will cause code assertions to fail if the data is aquired
    more than once.
    ArrayHandle is intended to be used within a scope limiting its use. For example:
    \code
GPUArray<int> gpu_array(100);
    {
    ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
    ... use h_handle.data ...
    }
    \endcode
    The actual raw pointer \a data should \b NOT be assumed to be the same after the handle is released.
    The pointer may in fact be re-allocated somewhere else after the handle is released and before the next handle
    is acquired.
    \ingroup data_structs
*/
template <class T>
class ArrayHandle
{
public:
    //! Aquires the data and sets \a m_data
    inline ArrayHandle(const GPUArray<T> &gpu_array, const access_location::Enum location = access_location::host,
                       const access_mode::Enum mode = access_mode::readwrite);
    //! Notifies the containing GPUArray that the handle has been released
    inline ~ArrayHandle();

    T *const data; //!< Pointer to data

private:
    const GPUArray<T> &m_gpu_array; //!< Reference to the GPUArray that owns \a data
};

#ifdef ENABLE_CUDA
//! Implementation of ArrayHandle using asynchronous copying between host and device
/*! This handle can be used to speed up access to the GPUArray data when
    accessing multiple buffers on the host AND the device.
    ArrayHandleAsync is asynchronous with respect to the host, i.e. multiple
    ArrayHandleAync objects maybe instantiated for multiple GPUArrays in a row, without
    incurring significant overhead for each of the handles.
    \warning Because ArrayHandleAsync uses asynchronous copying, however, array data is not
    guaranteed to be available on the host unless the device has been synchronized.
    Example usage:
    \code
GPUArray<int> gpu_array_1(100);
GPUArray<int> gpu_array_2(100);
    {
    ArrayHandle<int> h_handle_1(gpu_array_1, access_location::host, access_mode::readwrite);
    ArrayHandle<int> h_handle_2(gpu_array_2, access_location:::host, access_mode::readwrite);
    cudaDeviceSynchronize();
    ... use h_handle_1.data and h_handle_2.data ...
    }
    \endcode
*/
template <class T>
class ArrayHandleAsync
{
public:
    //! Aquires the data and sets \a m_data using asynchronous copies
    inline ArrayHandleAsync(const GPUArray<T> &gpu_array, const access_location::Enum location = access_location::host,
                            const access_mode::Enum mode = access_mode::readwrite);

    //! Notifies the containing GPUArray that the handle has been released
    virtual inline ~ArrayHandleAsync();

    T *const data; //!< Pointer to data

private:
    const GPUArray<T> &m_gpu_array; //!< Reference to the GPUArray that owns \a data
};
#endif

//! Class for managing an array of elements on the GPU mirrored to the CPU
/*!
GPUArray provides a template class for managing the majority of the GPU<->CPU memory usage patterns in
HOOMD. It represents a single array of elements which is present both on the CPU and GPU. Via
ArrayHandle, classes can access the array pointers through a handle for a short time. All needed
memory transfers from the host <-> device are handled by the class based on the access mode and
location specified when acquiring an ArrayHandle.
GPUArray is fairly advanced, C++ wise. It is a template class, so GPUArray's of floats, float4's,
uint2's, etc.. can be made. It comes with a copy constructor and = operator so you can (expensively)
pass GPUArray's around in arguments or overwite one with another via assignment (inexpensive swaps can be
performed with swap()). The ArrayHandle acquisition method guarantees that every aquired handle will be
released. About the only thing it \b doesn't do is prevent the user from writing to a pointer acquired
with a read only mode.
At a high level, GPUArray encapsulates a single flat data pointer \a T* \a data with \a num_elements
elements, and keeps a copy of this data on both the host and device. When accessing this data through
the construction of an ArrayHandle instance, the \a location (host or device) you wish to access the data
must be specified along with an access \a mode (read, readwrite, overwrite).
When the data is accessed in the same location it was last written to, the pointer is simply returned.
If the data is accessed in a different location, it will be copied before the pointer is returned.
When the data is accessed in the \a read mode, it is assumed that the data will not be written to and
thus there is no need to copy memory the next time the data is aquired somewhere else. Using the readwrite
mode specifies that the data is to be read and written to, necessitating possible copies to the desired location
before the data can be accessed and again before the next access. If the data is to be completely overwritten
\b without reading it first, then an expensive memory copy can be avoided by using the \a overwrite mode.
Data with both 1-D and 2-D representations can be allocated by using the appropriate constructor.
2-D allocated data is still just a flat pointer, but the row width is rounded up to a multiple of
16 elements to facilitate coalescing. The actual allocated width is accessible with getPitch(). Here
is an example of addressing element i,j in a 2-D allocated GPUArray.
\code
GPUArray<int> gpu_array(100, 200, m_exec_conf);
unsigned int pitch = gpu_array.getPitch();
ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
h_handle.data[i*pitch + j] = 5;
\endcode
A future modification of GPUArray will allow mirroring or splitting the data across multiple GPUs.
\ingroup data_structs
*/
template <class T>
class GPUArray
{
public:
    //! Constructs a NULL GPUArray
    GPUArray();
    //! Constructs a 1-D GPUArray
    GPUArray(unsigned int num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf);
    //! Constructs a 2-D GPUArray
    GPUArray(unsigned int width, unsigned int height, std::shared_ptr<const ExecutionConfiguration> exec_conf);
    //! Frees memory
    virtual ~GPUArray();

#ifdef ENABLE_CUDA
    //! Constructs a 1-D GPUArray
    GPUArray(unsigned int num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped);
    //! Constructs a 2-D GPUArray
    GPUArray(unsigned int width, unsigned int height, std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped);
#endif

    //! Copy constructor
    GPUArray(const GPUArray &from);
    //! = operator
    GPUArray &operator=(const GPUArray &rhs);

    //! Swap the pointers in two GPUArrays
    inline void swap(GPUArray &from);

    //! Swap the pointers of two equally sized GPUArrays
    inline void swap(GPUArray &from) const;

    //! Get the number of elements
    /*!
         - For 1-D allocated GPUArrays, this is the number of elements allocated.
         - For 2-D allocated GPUArrays, this is the \b total number of elements (\a pitch * \a height) allocated
        */
    unsigned int getNumElements() const
    {
        return m_num_elements;
    }

    //! Test if the GPUArray is NULL
    bool isNull() const
    {
        return (h_data == NULL);
    }

    //! Get the width of the allocated rows in elements
    /*!
         - For 2-D allocated GPUArrays, this is the total width of a row in memory (including the padding added for coalescing)
         - For 1-D allocated GPUArrays, this is the simply the number of elements allocated.
        */
    unsigned int getPitch() const
    {
        return m_pitch;
    }

    //! Get the number of rows allocated
    /*!
         - For 2-D allocated GPUArrays, this is the height given to the constructor
         - For 1-D allocated GPUArrays, this is the simply 1.
        */
    unsigned int getHeight() const
    {
        return m_height;
    }

    //! Resize the GPUArray
    /*! This method resizes the array by allocating a new array and copying over the elements
            from the old array. This is a slow process.
            Only data from the currently active memory location (gpu/cpu) is copied over to the resized
            memory area.
        */
    virtual void resize(unsigned int num_elements);

    //! Resize a 2D GPUArray
    virtual void resize(unsigned int width, unsigned int height);

protected:
    //! Clear memory starting from a given element
    /*! \param first The first element to clear
         */
    inline void memclear(unsigned int first = 0);

    //! Acquires the data pointer for use
    inline T *aquire(const access_location::Enum location, const access_mode::Enum mode
#ifdef ENABLE_CUDA
                     ,
                     bool async = false
#endif
                     ) const;

    //! Release the data pointer
    inline void release() const
    {
        m_acquired = false;
    }

private:
    mutable unsigned int m_num_elements; //!< Number of elements
    mutable unsigned int m_pitch;        //!< Pitch of the rows in elements
    mutable unsigned int m_height;       //!< Number of allocated rows

    mutable bool m_acquired;                     //!< Tracks whether the data has been aquired
    mutable data_location::Enum m_data_location; //!< Tracks the current location of the data
#ifdef ENABLE_CUDA
    mutable bool m_mapped; //!< True if we are using mapped memory
#endif

    // ok, this looks weird, but I want m_exec_conf to be protected and not have to go reorder all of the initializers
protected:
#ifdef ENABLE_CUDA
    mutable T *d_data; //!< Pointer to allocated device memory
#endif

    mutable T *h_data; //!< Pointer to allocated host memory

    mutable std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< execution configuration for working with CUDA
private:
    //! Helper function to allocate memory
    inline void allocate();
    //! Helper function to free memory
    inline void deallocate();

#ifdef ENABLE_CUDA
    //! Helper function to copy memory from the device to host
    inline void memcpyDeviceToHost(bool async) const;
    //! Helper function to copy memory from the host to device
    inline void memcpyHostToDevice(bool async) const;
#endif

    //! Helper function to resize host array
    inline T *resizeHostArray(unsigned int num_elements);

    //! Helper function to resize a 2D host array
    inline T *resize2DHostArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height);

    //! Helper function to resize device array
    inline T *resizeDeviceArray(unsigned int num_elements);

    //! Helper function to resize a 2D device array
    inline T *resize2DDeviceArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height);

    // need to be frineds of all the implementations of ArrayHandle and ArrayHandleAsync
    friend class ArrayHandle<T>;
#ifdef ENABLE_CUDA
    friend class ArrayHandleAsync<T>;
#endif
};

//******************************************
// ArrayHandle implementation
// *****************************************

/*! \param gpu_array GPUArray host to the pointer data
    \param location Desired location to access the data
    \param mode Mode to access the data with
*/
template <class T>
ArrayHandle<T>::ArrayHandle(const GPUArray<T> &gpu_array, const access_location::Enum location,
                            const access_mode::Enum mode) : data(gpu_array.aquire(location, mode)), m_gpu_array(gpu_array)
{
}

template <class T>
ArrayHandle<T>::~ArrayHandle()
{
    assert(m_gpu_array.m_acquired);
    m_gpu_array.m_acquired = false;
}

#ifdef ENABLE_CUDA
template <class T>
ArrayHandleAsync<T>::ArrayHandleAsync(const GPUArray<T> &gpu_array, const access_location::Enum location,
                                      const access_mode::Enum mode) : data(gpu_array.aquire(location, mode, true)), m_gpu_array(gpu_array)
{
}

template <class T>
ArrayHandleAsync<T>::~ArrayHandleAsync()
{
    m_gpu_array.m_acquired = false;
}
#endif

//******************************************
// GPUArray implementation
// *****************************************

template <class T>
GPUArray<T>::GPUArray() : m_num_elements(0), m_pitch(0), m_height(0), m_acquired(false), m_data_location(data_location::host),
#ifdef ENABLE_CUDA
                          m_mapped(false),
                          d_data(NULL),
#endif
                          h_data(NULL)
{
}

/*! \param num_elements Number of elements to allocate in the array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template <class T>
GPUArray<T>::GPUArray(unsigned int num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf) : m_num_elements(num_elements), m_pitch(num_elements), m_height(1), m_acquired(false), m_data_location(data_location::host),
#ifdef ENABLE_CUDA
                                                                                                            m_mapped(false),
                                                                                                            d_data(NULL),
#endif
                                                                                                            h_data(NULL),
                                                                                                            m_exec_conf(exec_conf)
{
    // allocate and clear memory
    allocate();
    memclear();
}

/*! \param width Width of the 2-D array to allocate (in elements)
    \param height Number of rows to allocate in the 2D array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template <class T>
GPUArray<T>::GPUArray(unsigned int width, unsigned int height, std::shared_ptr<const ExecutionConfiguration> exec_conf) : m_height(height), m_acquired(false), m_data_location(data_location::host),
#ifdef ENABLE_CUDA
                                                                                                                          m_mapped(false),
                                                                                                                          d_data(NULL),
#endif
                                                                                                                          h_data(NULL),
                                                                                                                          m_exec_conf(exec_conf)
{
    // make m_pitch the next multiple of 16 larger or equal to the given width
    m_pitch = (width + (16 - (width & 15)));

    // setup the number of elements
    m_num_elements = m_pitch * m_height;

    // allocate and clear memory
    allocate();
    memclear();
}

#ifdef ENABLE_CUDA
/*! \param num_elements Number of elements to allocate in the array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
    \param mapped True if we are using mapped-pinned memory
*/
template <class T>
GPUArray<T>::GPUArray(unsigned int num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped) : m_num_elements(num_elements), m_pitch(num_elements), m_height(1), m_acquired(false), m_data_location(data_location::host),
                                                                                                                         m_mapped(mapped),
                                                                                                                         d_data(NULL),
                                                                                                                         h_data(NULL),
                                                                                                                         m_exec_conf(exec_conf)
{
    // allocate and clear memory
    allocate();
    memclear();
}

/*! \param width Width of the 2-D array to allocate (in elements)
    \param height Number of rows to allocate in the 2D array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
    \param mapped True if we are using mapped-pinned memory
*/
template <class T>
GPUArray<T>::GPUArray(unsigned int width, unsigned int height, std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped) : m_height(height), m_acquired(false), m_data_location(data_location::host),
                                                                                                                                       m_mapped(mapped),
                                                                                                                                       d_data(NULL),
                                                                                                                                       h_data(NULL),
                                                                                                                                       m_exec_conf(exec_conf)
{
    // make m_pitch the next multiple of 16 larger or equal to the given width
    m_pitch = (width + (16 - (width & 15)));

    // setup the number of elements
    m_num_elements = m_pitch * m_height;

    // allocate and clear memory
    allocate();
    memclear();
}
#endif

template <class T>
GPUArray<T>::~GPUArray()
{
    deallocate();
}

template <class T>
GPUArray<T>::GPUArray(const GPUArray &from) : m_num_elements(from.m_num_elements), m_pitch(from.m_pitch),
                                              m_height(from.m_height), m_acquired(false), m_data_location(data_location::host),
#ifdef ENABLE_CUDA
                                              m_mapped(from.m_mapped),
                                              d_data(NULL),
#endif
                                              h_data(NULL),
                                              m_exec_conf(from.m_exec_conf)
{
    // allocate and clear new memory the same size as the data in from
    allocate();
    memclear();

    // copy over the data to the new GPUArray
    if (m_num_elements > 0)
    {
        ArrayHandle<T> h_handle(from, access_location::host, access_mode::read);
        memcpy(h_data.get(), h_handle.data, sizeof(T) * m_num_elements);
    }
}

template <class T>
GPUArray<T> &GPUArray<T>::operator=(const GPUArray &rhs)
{
    if (this != &rhs) // protect against invalid self-assignment
    {
        // sanity check
        assert(!m_acquired && !rhs.m_acquired);

        // free current memory
        deallocate();

        // copy over basic elements
        m_num_elements = rhs.m_num_elements;
        m_pitch = rhs.m_pitch;
        m_height = rhs.m_height;
        m_exec_conf = rhs.m_exec_conf;
#ifdef ENABLE_CUDA
        m_mapped = rhs.m_mapped;
#endif
        // initialize state variables
        m_data_location = data_location::host;

        // allocate and clear new memory the same size as the data in rhs
        allocate();
        memclear();

        // copy over the data to the new GPUArray
        if (m_num_elements > 0)
        {
            ArrayHandle<T> h_handle(rhs, access_location::host, access_mode::read);
            memcpy(h_data, h_handle.data, sizeof(T) * m_num_elements);
        }
    }

    return *this;
}

/*! \param from GPUArray to swap \a this with
    a.swap(b) will result in the equivalent of:
    \code
GPUArray c(a);
a = b;
b = c;
    \endcode
    But it will be done in a super-efficent way by just swapping the internal pointers, thus avoiding all the expensive
    memory deallocations/allocations and copies using the copy constructor and assignment operator.
*/
template <class T>
void GPUArray<T>::swap(GPUArray &from)
{
    // this may work, but really shouldn't be done when aquired
    assert(!m_acquired && !from.m_acquired);
    assert(&from != this);

    std::swap(m_num_elements, from.m_num_elements);
    std::swap(m_pitch, from.m_pitch);
    std::swap(m_height, from.m_height);
    std::swap(m_acquired, from.m_acquired);
    std::swap(m_data_location, from.m_data_location);
    std::swap(m_exec_conf, from.m_exec_conf);
#ifdef ENABLE_CUDA
    std::swap(d_data, from.d_data);
    std::swap(m_mapped, from.m_mapped);
#endif
    std::swap(h_data, from.h_data);
}

//! Swap the pointers of two GPUArrays (const version)
template <class T>
void GPUArray<T>::swap(GPUArray &from) const
{
    assert(!m_acquired && !from.m_acquired);
    assert(&from != this);

    std::swap(m_num_elements, from.m_num_elements);
    std::swap(m_pitch, from.m_pitch);
    std::swap(m_height, from.m_height);
    std::swap(m_exec_conf, from.m_exec_conf);
    std::swap(m_acquired, from.m_acquired);
    std::swap(m_data_location, from.m_data_location);
#ifdef ENABLE_CUDA
    std::swap(d_data, from.d_data);
    std::swap(m_mapped, from.m_mapped);
#endif
    std::swap(h_data, from.h_data);
}

/*! \pre m_num_elements is set
    \pre pointers are not allocated
    \post All memory pointers needed for GPUArray are allocated
*/
template <class T>
void GPUArray<T>::allocate()
{
    // don't allocate anything if there are zero elements
    if (m_num_elements == 0)
        return;

    // notify at a high level if a large allocation is about to occur
    if (m_num_elements > LARGEALLOCBYTES / (unsigned int)sizeof(T) && m_exec_conf)
    {
        // m_exec_conf->msg->notice(7) << "GPUArray is trying to allocate a very large (>4GB) amount of memory." << std::endl;
    }

#ifdef ENABLE_CUDA
    // we require mapped pinned memory
    if (m_mapped && m_exec_conf && !m_exec_conf->dev_prop.canMapHostMemory)
    {
        // if (m_exec_conf)
        //     m_exec_conf->msg->error() << "Device does not support mapped pinned memory." << std::endl
        //                               << std::endl;
        throw std::runtime_error("Error allocating GPUArray.");
    }
#endif

    // if (m_exec_conf)
    //     m_exec_conf->msg->notice(7) << "GPUArray: Allocating " << float(m_num_elements * sizeof(T)) / 1024.0f / 1024.0f << " MB" << std::endl;

    // sanity check
    assert(h_data == NULL);

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign((void **)&h_data, 32, m_num_elements * sizeof(T));
    if (retval != 0)
    {
        // if (m_exec_conf)
        //     m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
        throw std::runtime_error("Error allocating GPUArray.");
    }

#ifdef ENABLE_CUDA
    assert(d_data == NULL);
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
    {
        // register pointer for DMA
        cudaHostRegister(h_data, m_num_elements * sizeof(T), m_mapped ? cudaHostRegisterMapped : cudaHostRegisterDefault);

        // allocate and/or map host memory
        if (m_mapped)
        {
            cudaHostGetDevicePointer(&d_data, h_data, 0);
            CHECK_CUDA_ERROR();
        }
        else
        {
            cudaMalloc(&d_data, m_num_elements * sizeof(T));
            CHECK_CUDA_ERROR();
        }
    }
#endif
}

/*! \pre allocate() has been called
    \post All allocated memory is freed
*/
template <class T>
void GPUArray<T>::deallocate()
{
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;

    // sanity check
    assert(!m_acquired);
    assert(h_data);

    // free memory

#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
    {
        assert(d_data);
        cudaHostUnregister(h_data);
        CHECK_CUDA_ERROR();

        if (!m_mapped)
        {
            cudaFree(d_data);
            CHECK_CUDA_ERROR();
        }
    }
#endif

    free(h_data);

    // set pointers to NULL
    h_data = NULL;
#ifdef ENABLE_CUDA
    d_data = NULL;
#endif
}

/*! \pre allocate() has been called
    \post All allocated memory is set to 0
*/
template <class T>
void GPUArray<T>::memclear(unsigned int first)
{
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;

    assert(h_data);
    assert(first < m_num_elements);

    // clear memory
    memset((void *)(h_data + first), 0, sizeof(T) * (m_num_elements - first));

#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
    {
        assert(d_data);
        if (!m_mapped)
            cudaMemset(d_data + first, 0, (m_num_elements - first) * sizeof(T));
    }
#endif
}

#ifdef ENABLE_CUDA
/*! \post All memory on the device is copied to the host array
*/
template <class T>
void GPUArray<T>::memcpyDeviceToHost(bool async) const
{
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;

    if (m_mapped)
    {
        // if we are using mapped pinned memory, no need to copy, only synchronize
        if (!async)
            cudaDeviceSynchronize();
        return;
    }

    // if (m_exec_conf)
    //     m_exec_conf->msg->notice(8) << "GPUArray: Copying " << float(m_num_elements * sizeof(T)) / 1024.0f / 1024.0f << " MB device->host" << std::endl;
    if (async)
        cudaMemcpyAsync(h_data, d_data, sizeof(T) * m_num_elements, cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(h_data, d_data, sizeof(T) * m_num_elements, cudaMemcpyDeviceToHost);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
}

/*! \post All memory on the host is copied to the device array
*/
template <class T>
void GPUArray<T>::memcpyHostToDevice(bool async) const
{
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;

    if (m_mapped)
    {
        // if we are using mapped pinned memory, no need to copy
        // rely on CUDA's implicit synchronization
        return;
    }

    // if (m_exec_conf)
    //     m_exec_conf->msg->notice(8) << "GPUArray: Copying " << float(m_num_elements * sizeof(T)) / 1024.0f / 1024.0f << " MB host->device" << std::endl;
    if (async)
        cudaMemcpyAsync(d_data, h_data, sizeof(T) * m_num_elements, cudaMemcpyHostToDevice);
    else
        cudaMemcpy(d_data, h_data, sizeof(T) * m_num_elements, cudaMemcpyHostToDevice);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
}
#endif

/*! \param location Desired location to access the data
    \param mode Mode to access the data with
    \param async True if array copying should be done async
    aquire() is the workhorse of GPUArray. It tracks the internal state variable \a data_location and
    performs all host<->device memory copies as needed during the state changes given the
    specified access mode and location where the data is to be acquired.
    aquire() cannot be directly called by the user class. Data must be accessed through ArrayHandle.
*/
template <class T>
T *GPUArray<T>::aquire(const access_location::Enum location, const access_mode::Enum mode
#ifdef ENABLE_CUDA
                       ,
                       bool async
#endif
                       ) const
{
    // sanity check
    assert(!m_acquired);
    m_acquired = true;

    // base case - handle acquiring a NULL GPUArray by simply returning NULL to prevent any memcpys from being attempted
    if (isNull())
        return NULL;

    if (m_exec_conf)
        m_exec_conf->setGPUDevice();

    // first, break down based on where the data is to be acquired
    if (location == access_location::host)
    {
        // then break down based on the current location of the data
        if (m_data_location == data_location::host)
        {
            // the state stays on the host regardles of the access mode
            return h_data;
        }
#ifdef ENABLE_CUDA
        else if (m_data_location == data_location::hostdevice)
        {
            // finally perform the action baed on the access mode requested
            if (mode == access_mode::read) // state stays on hostdevice
                m_data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite) // state goes to host
                m_data_location = data_location::host;
            else if (mode == access_mode::overwrite) // state goes to host
                m_data_location = data_location::host;
            else
            {
                // if (m_exec_conf)
                //     m_exec_conf->msg->error() << "Invalid access mode requested" << std::endl;
                throw std::runtime_error("Error acquiring data");
            }

            return h_data;
        }
        else if (m_data_location == data_location::device)
        {
            // finally perform the action baed on the access mode requested
            if (mode == access_mode::read)
            {
                // need to copy data from the device to the host
                memcpyDeviceToHost(async);
                // state goes to hostdevice
                m_data_location = data_location::hostdevice;
            }
            else if (mode == access_mode::readwrite)
            {
                // need to copy data from the device to the host
                memcpyDeviceToHost(async);
                // state goes to host
                m_data_location = data_location::host;
            }
            else if (mode == access_mode::overwrite)
            {
                // no need to copy data, it will be overwritten
                // state goes to host
                m_data_location = data_location::host;
            }
            else
            {
                // if (m_exec_conf)
                //     m_exec_conf->msg->error() << "Invalid access mode requested" << std::endl;
                throw std::runtime_error("Error acquiring data");
            }

            return h_data;
        }
#endif
        else
        {
            // if (m_exec_conf)
            //     m_exec_conf->msg->error() << "Invalid data location state" << std::endl;
            throw std::runtime_error("Error acquiring data");
            return NULL;
        }
    }
#ifdef ENABLE_CUDA
    else if (location == access_location::device)
    {
        // check that a GPU is actually specified
        if (!m_exec_conf)
        {
            throw std::runtime_error("Requesting device aquire, but we have no execution configuration");
        }
        if (!m_exec_conf->isCUDAEnabled())
        {
            // m_exec_conf->msg->error() << "Requesting device aquire, but no GPU in the Execution Configuration" << std::endl;
            throw std::runtime_error("Error acquiring data");
        }

        // then break down based on the current location of the data
        if (m_data_location == data_location::host)
        {
            // finally perform the action baed on the access mode requested
            if (mode == access_mode::read)
            {
                // need to copy data to the device
                memcpyHostToDevice(async);
                // state goes to hostdevice
                m_data_location = data_location::hostdevice;
            }
            else if (mode == access_mode::readwrite)
            {
                // need to copy data to the device
                memcpyHostToDevice(async);
                // state goes to device
                m_data_location = data_location::device;
            }
            else if (mode == access_mode::overwrite)
            {
                // no need to copy data to the device, it is to be overwritten
                // state goes to device
                m_data_location = data_location::device;
            }
            else
            {
                // m_exec_conf->msg->error() << "Invalid access mode requested" << std::endl;
                throw std::runtime_error("Error acquiring data");
            }

            return d_data;
        }
        else if (m_data_location == data_location::hostdevice)
        {
            // finally perform the action baed on the access mode requested
            if (mode == access_mode::read) // state stays on hostdevice
                m_data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite) // state goes to device
                m_data_location = data_location::device;
            else if (mode == access_mode::overwrite) // state goes to device
                m_data_location = data_location::device;
            else
            {
                // m_exec_conf->msg->error() << "Invalid access mode requested" << std::endl;
                throw std::runtime_error("Error acquiring data");
            }
            return d_data;
        }
        else if (m_data_location == data_location::device)
        {
            // the stat stays on the device regardless of the access mode
            return d_data;
        }
        else
        {
            // m_exec_conf->msg->error() << "Invalid data_location state" << std::endl;
            throw std::runtime_error("Error acquiring data");
            return NULL;
        }
    }
#endif
    else
    {
        // if (m_exec_conf)
        //     m_exec_conf->msg->error() << "Invalid location requested" << std::endl;
        throw std::runtime_error("Error acquiring data");
        return NULL;
    }
}

/*! \post Memory on the host is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a pointer to the newly allocated memory area
*/
template <class T>
T *GPUArray<T>::resizeHostArray(unsigned int num_elements)
{
    // if not allocated, do nothing
    if (isNull())
        return NULL;

    // allocate resized array
    T *h_tmp = NULL;

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign((void **)&h_tmp, 32, num_elements * sizeof(T));
    if (retval != 0)
    {
        // if (m_exec_conf)
        //     m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
        throw std::runtime_error("Error allocating GPUArray.");
    }

#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
    {
        cudaHostRegister(h_tmp, num_elements * sizeof(T), m_mapped ? cudaHostRegisterMapped : cudaHostRegisterDefault);
    }
#endif
    // clear memory
    memset((void *)h_tmp, 0, sizeof(T) * num_elements);

    // copy over data
    unsigned int num_copy_elements = m_num_elements > num_elements ? num_elements : m_num_elements;
    memcpy((void *)h_tmp, (void *)h_data, sizeof(T) * num_copy_elements);

    // free old memory location
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
    {
        cudaHostUnregister(h_data);
        CHECK_CUDA_ERROR();
    }
#endif

    free(h_data);
    h_data = h_tmp;

#ifdef ENABLE_CUDA
    // update device pointer
    if (m_mapped)
        cudaHostGetDevicePointer(&d_data, h_data, 0);
#endif

    return h_data;
}

/*! \post Memory on the host is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a pointer to the newly allocated memory area
*/
template <class T>
T *GPUArray<T>::resize2DHostArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height)
{
    // allocate resized array
    T *h_tmp = NULL;

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    unsigned int size = new_pitch * new_height * sizeof(T);
    int retval = posix_memalign((void **)&h_tmp, 32, size);
    if (retval != 0)
    {
        // if (m_exec_conf)
        //     m_exec_conf->msg->error() << "Error allocating aligned memory" << std::endl;
        throw std::runtime_error("Error allocating GPUArray.");
    }

#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
    {
        cudaHostRegister(h_tmp, size, cudaHostRegisterDefault);
    }
#endif

    // clear memory
    memset((void *)h_tmp, 0, sizeof(T) * new_pitch * new_height);

    // copy over data
    // every column is copied separately such as to align with the new pitch
    unsigned int num_copy_rows = height > new_height ? new_height : height;
    unsigned int num_copy_columns = pitch > new_pitch ? new_pitch : pitch;
    for (unsigned int i = 0; i < num_copy_rows; i++)
        memcpy((void *)(h_tmp + i * new_pitch), (void *)(h_data + i * pitch), sizeof(T) * num_copy_columns);

        // free old memory location
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
    {
        cudaHostUnregister(h_data);
        CHECK_CUDA_ERROR();
    }
#endif

    free(h_data);
    h_data = h_tmp;

#ifdef ENABLE_CUDA
    // update device pointer
    if (m_mapped)
        cudaHostGetDevicePointer(&d_data, h_data, 0);
#endif

    return h_data;
}

/*! \post Memory on the device is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a device pointer to the newly allocated memory area
*/
template <class T>
T *GPUArray<T>::resizeDeviceArray(unsigned int num_elements)
{
#ifdef ENABLE_CUDA
    if (m_mapped)
        return NULL;

    // allocate resized array
    T *d_tmp;
    cudaMalloc(&d_tmp, num_elements * sizeof(T));
    CHECK_CUDA_ERROR();

    assert(d_tmp);

    // clear memory
    cudaMemset(d_tmp, 0, num_elements * sizeof(T));
    CHECK_CUDA_ERROR();

    // copy over data
    unsigned int num_copy_elements = m_num_elements > num_elements ? num_elements : m_num_elements;
    cudaMemcpy(d_tmp, d_data, sizeof(T) * num_copy_elements, cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR();

    // free old memory location
    cudaFree(d_data);
    CHECK_CUDA_ERROR();

    d_data = d_tmp;
    return d_data;
#else
    return NULL;
#endif
}

/*! \post Memory on the device is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a device pointer to the newly allocated memory area
*/
template <class T>
T *GPUArray<T>::resize2DDeviceArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height)
{
#ifdef ENABLE_CUDA
    if (m_mapped)
        return NULL;

    // allocate resized array
    T *d_tmp;
    cudaMalloc(&d_tmp, new_pitch * new_height * sizeof(T));
    CHECK_CUDA_ERROR();

    assert(d_tmp);

    // clear memory
    cudaMemset(d_tmp, 0, new_pitch * new_height * sizeof(T));
    CHECK_CUDA_ERROR();

    // copy over data
    // every column is copied separately such as to align with the new pitch
    unsigned int num_copy_rows = height > new_height ? new_height : height;
    unsigned int num_copy_columns = pitch > new_pitch ? new_pitch : pitch;

    for (unsigned int i = 0; i < num_copy_rows; i++)
    {
        cudaMemcpy(d_tmp + i * new_pitch, d_data + i * pitch, sizeof(T) * num_copy_columns, cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR();
    }

    // free old memory location
    cudaFree(d_data);
    CHECK_CUDA_ERROR();

    d_data = d_tmp;
    return d_data;
#else
    return NULL;
#endif
}

/*! \param num_elements new size of array
 *
 * \warning An array can be expanded or shrunk, depending on the parameters supplied.
 *          It is the responsibility of the caller to ensure that no data is inadvertently lost when
 *          reducing the size of the array.
*/
template <class T>
void GPUArray<T>::resize(unsigned int num_elements)
{
    assert(!m_acquired);
    assert(num_elements > 0);

    // if not allocated, simply allocate
    if (isNull())
    {
        m_num_elements = num_elements;
        allocate();
        return;
    };

    // notify at a high level if a large allocation is about to occur
    if (m_num_elements > LARGEALLOCBYTES / (unsigned int)sizeof(T) && m_exec_conf)
    {
        // m_exec_conf->msg->notice(7) << "GPUArray is trying to allocate a very large (>4GB) amount of memory." << std::endl;
    }

    // if (m_exec_conf)
    //     m_exec_conf->msg->notice(7) << "GPUArray: Resizing to " << float(num_elements * sizeof(T)) / 1024.0f / 1024.0f << " MB" << std::endl;

    resizeHostArray(num_elements);
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        resizeDeviceArray(num_elements);
#endif
    m_num_elements = num_elements;
    m_pitch = num_elements;
}

/*! \param width new width of array
*   \param height new height of array
*
*   \warning An array can be expanded or shrunk, depending on the parameters supplied.
*   It is the responsibility of the caller to ensure that no data is inadvertently lost when
*   reducing the size of the array.
*/
template <class T>
void GPUArray<T>::resize(unsigned int width, unsigned int height)
{
    assert(!m_acquired);

    // make m_pitch the next multiple of 16 larger or equal to the given width
    unsigned int new_pitch = (width + (16 - (width & 15)));

    unsigned int num_elements = new_pitch * height;
    assert(num_elements > 0);

    // if not allocated, simply allocate
    if (isNull())
    {
        m_num_elements = num_elements;
        allocate();
        m_pitch = new_pitch;
        m_height = height;
        return;
    };

    // notify at a high level if a large allocation is about to occur
    if (m_num_elements > LARGEALLOCBYTES / (unsigned int)sizeof(T) && m_exec_conf)
    {
        // m_exec_conf->msg->notice(7) << "GPUArray is trying to allocate a very large (>4GB) amount of memory." << std::endl;
    }

    resize2DHostArray(m_pitch, new_pitch, m_height, height);
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        resize2DDeviceArray(m_pitch, new_pitch, m_height, height);
#endif
    m_num_elements = num_elements;

    m_height = height;
    m_pitch = new_pitch;
    m_num_elements = m_pitch * m_height;
}
#endif