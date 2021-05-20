#pragma once

#include <cuda_runtime.h>

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

inline void CUDA_KERNEL_CHECK() {
    err = cudaGetLastError();
    if  (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    } else {
        cerr << "No kernel error detected" << endl;
    }
}
