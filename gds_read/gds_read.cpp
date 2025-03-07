#include <torch/extension.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <future> 
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "cufile.h"
#include "cufile_sample_utils.h"

namespace py = pybind11;

torch::Tensor readfile(const std::string& filepath, int64_t offset, int64_t len) {
    int fd;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;
    void* devPtr = nullptr;
    ssize_t ret = -1;

    // Open the file for reading
    fd = open(filepath.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    // Register the file with cuFile
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        throw std::runtime_error("Failed to register file: " + std::string(cuFileGetErrorString(status)));
    }

    // Allocate GPU memory
    check_cudaruntimecall(cudaMalloc(&devPtr, len));

    // Read data from file into GPU memory
    ret = cuFileRead(cf_handle, devPtr, len, offset, 0);
    if (ret < 0) {
        cuFileHandleDeregister(cf_handle);
        close(fd);
        cudaFree(devPtr);
        throw std::runtime_error("Failed to read file: " + std::string(cuFileGetErrorString(ret)));
    }

    // Deregister and close the file
    cuFileHandleDeregister(cf_handle);
    close(fd);

    // Create a tensor from the GPU memory
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
    auto tensor = torch::from_blob(devPtr, {len}, options).clone();

    // Free the GPU memory
    check_cudaruntimecall(cudaFree(devPtr));

    return tensor;
}

std::future<torch::Tensor> readfile_async(const std::string& filepath, int64_t offset, int64_t len) {
    return std::async(std::launch::async, readfile, filepath, offset, len);
}

py::object readfile_async_py(const std::string& filepath, int64_t offset, int64_t len) {
    py::gil_scoped_release release;  // Release GIL
    auto future = readfile_async(filepath, offset, len);
    auto result = future.get();
    py::gil_scoped_acquire acquire;  // Re-acquire GIL
    return py::cast(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("readfile", &readfile, "Read file using GDS (synchronous)");
    m.def("readfile_async", &readfile_async_py, "Read file using GDS (asynchronous)");
}
