#include "grid.cuh"

#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gridLabelKernel(uint *dev_pt_ids, uint *dev_grid_labels,
                                float *dev_coords,
                                float min_x, float min_y, float side_len,
                                uint grid_x_size, int num) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < num; idx += blockDim.x) {
        dev_pt_ids[idx] = idx;
        uint x = (uint) ((dev_coords[2*idx] - min_x) / side_len);
        uint y = (uint) ((dev_coords[2*idx+1] - min_y) / side_len);
        uint label = y*grid_x_size + x;
        dev_grid_labels[idx] = label;
    }
}
__global__ void gridMarkCoreCells(uint *d_index_counts, uint unique_key_count,
                                  uint *d_values, bool *isCore, uint min_points) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < unique_key_count; idx += blockDim.x) {
        uint start = d_index_counts[2*idx];
        uint length = d_index_counts[2*idx + 1];
        if (length >= min_points) {
            for (uint i = start; i < start + length; i++) {
                isCore[d_values[i]] = true;
            }
        }
    }
}

// Always called with one block since key_count <= 21
__global__ void gridCheckCore(float *dev_coords, uint *d_index_counts,
                              uint key_count, uint *d_values, bool *d_isCore,
                              uint min_points, float EPS_SQ, float x, float y,
                              int pt_idx) {
    __shared__ int count;
    if (threadIdx.x == 0)
        count = 0;
    __syncthreads();
    uint start = d_index_counts[2*threadIdx.x];
    uint length = d_index_counts[2*threadIdx.x+1];
    for (uint i = start; i < start + length && count < min_points; i++) {
        float x2 = dev_coords[d_values[i]*2];
        float y2 = dev_coords[d_values[i]*2 + 1];
        if ((x2 - x) * (x2 - x) + (y2 - y) * (y2 - y) <= EPS_SQ) {
            atomicAdd(&count, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && count >= min_points) {
        d_isCore[pt_idx] = true;
    }        
}
    
    

void callGridLabelKernel(uint blocks, uint threadsPerBlock,
                         uint *dev_pt_ids, uint *dev_grid_labels,
                         float *dev_coords,
                         float min_x, float min_y, float side_len,
                         uint grid_x_size, int num) {
    gridLabelKernel<<<blocks, threadsPerBlock>>>(dev_pt_ids, dev_grid_labels,
                                                 dev_coords, min_x, min_y,
                                                 side_len, grid_x_size, num);
    CUDA_KERNEL_CHECK();
}

void callGridMarkCoreCells(uint blocks, uint threadsPerBlock,
                           uint *d_index_counts, uint unique_key_count,
                           uint *d_values, bool *isCore, uint min_points) {
    gridMarkCoreCells<<<blocks, threadsPerBlock>>>(
        d_index_counts, unique_key_count, d_values, isCore, min_points);
    CUDA_KERNEL_CHECK();
}

void callGridCheckCore(float *dev_coords, uint *d_index_counts,
                       uint key_count, uint *d_values, bool *d_isCore,
                       uint min_points, float EPS_SQ, float x, float y,
                       int pt_idx) {
    gridCheckCore<<<1, key_count>>>(dev_coords, d_index_counts,
                                    key_count, d_values, d_isCore, min_points,
                                    EPS_SQ, x, y, pt_idx);
    CUDA_KERNEL_CHECK();
}
