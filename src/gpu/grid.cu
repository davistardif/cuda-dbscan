#include "grid.cuh"

#include "cuda_utils.hpp"

#include <cuda_runtime.h>

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
