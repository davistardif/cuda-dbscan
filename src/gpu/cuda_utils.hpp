#pragma once

#include <cuda_runtime.h>
#include "cudpp.h"
#include <fstream>
#include <iostream>

using namespace std;

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define CUDPP_CALL(ans) { cudppAssert((ans), __FILE__, __LINE__); }
inline void cudppAssert(CUDPPResult code, const char *file, int line, bool abort=true)
{
    if (code != CUDPP_SUCCESS) 
    {
        fprintf(stderr,"CUDPP Error at: %s %d\n", file, line);
        exit(code);
    }
}

inline void CUDA_KERNEL_CHECK() {
    cudaError err = cudaGetLastError();
    if  (cudaSuccess != err){
        cerr << "Kernel error: " << cudaGetErrorString(err) << endl;
    } 
}
