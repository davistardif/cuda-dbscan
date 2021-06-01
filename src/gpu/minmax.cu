#include "minmax.cuh"

#include <cuda_runtime.h>

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


/*
 Compute the maximum x value and maximum y value in an xy coordinate array
 
 Coords is assumed to be a striped array (i.e. x values at even indices,
 y values at odd indices). 
 
 max_x and max_y are output parameters and should be initialized with an
 x and y value or suitably small value that will be less than the max
 
 length should be the size of the coordinate array (i.e. 2 times the number
 of (x,y) pairs)

 blockDim.x must be even for the kernel to work correctly
*/
__global__ void cudaMaxXYKernel(float *coords, int length, float *max_x,
                                float *max_y) {
    extern __shared__ float sdata[];
    uint tid = threadIdx.x;
    uint i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    for (; i < length; i += gridDim.x * blockDim.x * 2) {
        if (i + blockDim.x < length) {
            // if possible, each thread compares two values and
            // moves the larger into shared memory
            sdata[tid] = max(coords[i], coords[i + blockDim.x]);
        }
        else {
            // in case we are near the end of the array, one value
            // is copied into shared memory
            sdata[tid] = coords[i];
        }
        // allow all of shared mem to be populated
        __syncthreads();
        for (uint s = blockDim.x / 2; s > 1; s >>= 1) {
            // In each iteration, s threads compute a maximum and move it
            // to the left. Thus each iteration only has to consider the first
            // 2s values in shared memory. The stride is maximized at each
            // step to prevent bank conflicts 
            if (tid < s) {
                sdata[tid] = max(sdata[tid], sdata[tid + s]);
            }
            // at each stage of reduction, need to wait for all threads
            __syncthreads();
        }
        /* After reduction completes, max x value is position 0 in shared mem,
           and max y value is at position 1, so we use atomic max to put
           these values into the out params
        */
        if (tid == 0) {
            atomicMax(max_x, sdata[0]);
        }
        else if (tid == 1) {
            atomicMax(max_y, sdata[1]);
        }
    }
}

// See documentation for cudaMaxXYKernel above
__global__ void cudaMinXYKernel(float *coords, int length, float *min_x,
                                float *min_y) {
    extern __shared__ float sdata[];
    uint tid = threadIdx.x;
    uint i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    for (; i < length; i += gridDim.x * blockDim.x * 2) {
        if (i + blockDim.x < length) {
            sdata[tid] = min(coords[i], coords[i + blockDim.x]);
        }
        else {
            sdata[tid] = coords[i];
        }
        __syncthreads();
        for (uint s = blockDim.x / 2; s > 1; s >>= 1) {
            if (tid < s) {
                sdata[tid] = min(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicMin(min_x, sdata[0]);
        }
        else if (tid == 1) {
            atomicMin(min_y, sdata[1]);
        }
    }
}

void cudaCallMaxXYKernel(const unsigned int blocks,
                         const unsigned int threadsPerBlock,
                         float *coords, int length, float *max_x, float *max_y) {
    cudaMaxXYKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        coords, length, max_x, max_y);
}

void cudaCallMinXYKernel(const unsigned int blocks,
                         const unsigned int threadsPerBlock,
                         float *coords, int length, float *min_x, float *min_y) {
    cudaMinXYKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        coords, length, min_x, min_y);
}
