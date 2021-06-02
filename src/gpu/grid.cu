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
            printf("Core cell with %d points\n", length);
            for (uint i = start; i < start + length; i++) {
                isCore[d_values[i]] = true;
            }
        }
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
