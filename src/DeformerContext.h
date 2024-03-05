#ifndef OCL_DEFORMER_CONTEXT_H
#define OCL_DEFORMER_CONTEXT_H

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>

#include <iostream>
#include "opencl.hpp"

namespace peasyocl {

/**
 * @brief A handle to set arguments in a kernel source
 * 
 */
struct KernelHandle {
    cl::Kernel kernel;
    std::unordered_map<std::string, int> arguments;
    std::unordered_map<std::string, cl::Buffer> buffers;
    
    cl::CommandQueue* queue;
    cl::Context* context;

    int argCount = 0;

    /**
     * @brief Add argument to kernel and create buffer
     * 
     * @tparam T 
     * @param flags Flags passed to the cl::Buffer object
     * @param name Name associated with this argument
     * @param elems Number of elements
     * @param data Data of type T
     * @return int Success
     */
    template<typename T>
    int AddArgument(cl_mem_flags flags, const std::string& name, const size_t& elems, T* data);
    
    /**
     * @brief Add argument to kernel and create buffer
     * 
     * @tparam T 
     * @param flags Flags passed to the cl::Buffer object
     * @param name Name associated with this argument
     * @param elems Number of elements
     * @param data Data of type const T&
     * @return int Success
     */
    template<typename T>
    int AddArgument(cl_mem_flags flags, const std::string& name, const size_t& elems, const T& data);

        /**
     * @brief Add argument to kernel with option to create buffer. Allows one to create an empty buffer or argument
     * with no buffer attached
     * 
     * @tparam T Bool
     * @param flags Flags passed to the cl::Buffer object
     * @param name Name associated with this argument
     * @param elems Number of elements
     * @param createBuffer Whether to create an empty buffer. Defaults to true
     * @return int Success
     */
    template<typename T>
    int AddArgument(cl_mem_flags flags, const std::string& name, const size_t& elems, const bool createBuffer=true);

    /**
     * @brief Set the Argument at index argIndex to data
     * 
     * @tparam mem 
     * @tparam T 
     * @param argIndex Int in to the arguments map, i.e what argument to set
     * @param data Of type T* to set
     * @return int 
     */
    template<typename mem, typename T>
    int SetArgument(const int argIndex, T* data);

    /**
     * @brief Set the Argument at index argIndex to data
     * 
     * @tparam mem 
     * @tparam T 
     * @param argIndex Int in to the arguments map, i.e what argument to set
     * @param data Of type const T& to set
     * @return int 
     */
    template<typename T>
    int SetArgument(const int argIndex, const T& data);

    /**
     * @brief Set the Argument with name to data
     * 
     * @tparam mem 
     * @tparam T 
     * @param name The name of the argument to set
     * @param data Of type T* to set
     * @return int 
     */
    template<typename mem, typename T>
    int SetArgument(const std::string& name, T* data);

    /**
     * @brief Set the Argument with name to data
     * 
     * @tparam mem 
     * @tparam T 
     * @param name The name of the argument to set
     * @param data Of type const T& to set
     * @return int 
     */
    template<typename T>
    int SetArgument(const std::string& name, const T& data);

    /**
     * @brief Read data from buffer
     * 
     * @tparam T 
     * @param data Result data to write to
     * @param buffer Bufer object to read from
     * @param elems Number of elements to read
     * @return int 
     */
    template<typename T>
    int ReadBufferData(T* data, cl::Buffer buffer, const size_t elems);

    /**
     * @brief Set the data of buffer
     * 
     * @tparam T 
     * @param data Of type T*
     * @param buffer Buffer object to set
     * @param elems Number of elements to read
     * @return int 
     */
    template<typename T>
    int SetBufferData(T* data, cl::Buffer buffer, const size_t elems);

    /**
     * @brief Read the data of buffer with name
     * 
     * @tparam T 
     * @param data
     * @param bufferName Buffer to set. Name is associated with the name created with AddArgument function
     * @param elems Number of elements to read
     * @return int 
     */
    template<typename T>
    int ReadBufferData(T* data, const std::string& bufferName, const size_t elems);
    
    /**
     * @brief Set the data of buffer with name
     * 
     * @tparam T 
     * @param data Of type T*
     * @param bufferName Buffer to set. Name is associated with the name created with AddArgument function
     * @param elems Number of elements to read
     * @return int 
     */
    template<typename T>
    int SetBufferData(T* data, const std::string& bufferName, const size_t elems);
};

/**
 * @brief Singleton class to handle the opencl context
 * 
 */
class DeformerContext {
public:
    int Init();
    int Build();

    /**
     * @brief Get the Instance of the DeformerContext singleton
     * 
     * @return DeformerContext* 
     */
    static DeformerContext* getInstance() {
        static DeformerContext* ctx = new DeformerContext();
        return ctx;
    }

    /**
     * @brief Add a .cl or .ocl file for executing
     * 
     * @param fileName Name of opencl kernel file. Should be visible in OCL_KERNEL_PATHS environment
     * @return int 
     */
    int AddSource(const std::string& fileName);

    /**
     * @brief Load a kernel from the sources added. Returns a KernelHandle which is used to access kernel specific arguments.
     * 
     * @param kernelName Looks for kernels inside any added sources 
     * @param key Override internal key name for this kernel for storage inside the _kernels map
     * @param err 
     * @return KernelHandle* 
     */
    KernelHandle* AddKernel(const std::string& kernelName, const std::string& key="", int* err=nullptr);

    /**
     * @brief Get a pointer to the KernelHandle with name
     * 
     * @param name 
     * @return KernelHandle* 
     */
    KernelHandle* GetKernelHandle(const std::string& name);

    /**
     * @brief Check whether the specific kernel exists
     * 
     * @param name Name of kernel
     * @return true If kernel exists
     * @return false If kernel does not exist
     */
    bool HasKernel(const std::string& name) {return _kernels.find(name) != _kernels.end();}

    /**
     * @brief Execute a kernel with name kernelName
     * 
     * @param global Global workgroup size
     * @param kernelName Name of kernel to execute
     * @return int 
     */
    int Execute(const size_t& global, const std::string& kernelName);

    /**
     * @brief Flush and finish the queue
     * 
     */
    void Finish();

    // Is true once everything is initialized
    bool initialized = false;
    bool built = false;

private:
    DeformerContext() = default;
    DeformerContext(const DeformerContext&) = delete;
    DeformerContext(DeformerContext&&) = delete;
    DeformerContext& operator = (const DeformerContext&) = delete;
    DeformerContext& operator = (DeformerContext&&) = delete;

    cl::string _LoadShader(const std::string_view& fileName, int* err);

    cl::vector<cl::string>  _kernelCodes;

    cl::Context                         _context;
    cl::CommandQueue                    _queue;
    cl::Program                         _program;
    std::map<cl::string, KernelHandle>  _kernels;
    cl::Device                          _device;

    int _argumentsSize = 0;
};

template<typename T>
inline int KernelHandle::AddArgument(cl_mem_flags flags, const std::string& name, const size_t& elems, T* data) {
    cl::Buffer d_data(*context, flags, sizeof(T) * elems);

    if (data != nullptr) {
        SetBufferData(data, d_data, elems);
    }

    SetArgument<cl_mem, cl::Buffer>(argCount, &d_data);
    buffers.insert({name, d_data});

    arguments.insert({name, argCount});
    argCount++;
    return 0;
}

template<typename T>
inline int KernelHandle::AddArgument(cl_mem_flags flags, const std::string& name, const size_t& elems, const T& data) {
    cl::Buffer d_data(*context, flags, sizeof(T) * elems);

    SetBufferData(&data, d_data, elems);

    SetArgument<cl_mem, cl::Buffer>(argCount, &d_data);
    buffers.insert({name, d_data});

    arguments.insert({name, argCount});
    argCount++;
    return 0;
}

template<typename T>
inline int KernelHandle::AddArgument(cl_mem_flags flags, const std::string& name, const size_t& elems, const bool createBuffer) {
    if (createBuffer) {
        return AddArgument<T>(flags, name, elems, nullptr);
    }
    SetArgument<T>(argCount, (T*)nullptr);
    arguments.insert({name, argCount});
    argCount++;
    return 0;
}

template<typename mem, typename T>
inline int KernelHandle::SetArgument(const int argIndex, T* data) {
    kernel.setArg(argIndex, sizeof(mem), data);
    return 0;
}

template<typename T>
inline int KernelHandle::SetArgument(const int argIndex, const T& data) {
    kernel.setArg<T>(argIndex, data);
    return 0;
}

template<typename mem, typename T>
inline int KernelHandle::SetArgument(const std::string& name, T* data) {
    SetArgument(arguments[name], data);
    return 0;
}

template<typename T>
inline int KernelHandle::SetArgument(const std::string& name, const T& data) {
    SetArgument(arguments[name], data);
    return 0;
}

template<typename T>
inline int KernelHandle::SetBufferData(T* data, cl::Buffer buffer, const size_t elems) {
    cl_int err = queue->enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(T) * elems, data);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write data to source array!\n");
        return 1;
    }
    return 0;
}

template<typename T>
inline int KernelHandle::SetBufferData(T* data, const std::string& bufferName, const size_t elems) {
    if (arguments.find(bufferName) == arguments.end()) {
        printf("Error: Buffer %s is not recognized!\n", bufferName.c_str());
        return 1;
    }
    return SetBufferData(data, buffers[bufferName], elems);
}

template<typename T>
inline int KernelHandle::ReadBufferData(T* data, cl::Buffer buffer, const size_t elems) {
    cl_int err = queue->enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T) * elems, data);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        return 1;
    }
    return 0;
}

template<typename T>
inline int KernelHandle::ReadBufferData(T* data, const std::string& bufferName, const size_t elems) {
    return ReadBufferData(data, buffers[bufferName], elems);
}

}

#endif