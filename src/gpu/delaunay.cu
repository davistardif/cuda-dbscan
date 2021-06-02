#include "delaunay.cuh"

#include "clustering.hpp"
#include "point_set.hpp"
#include "cuda_utils.hpp"
#include "minmax.cuh"
#include "grid.cuh"

#include "cudpp.h"
#include "cudpp_hash.h"
#include "cudpp_config.h"

#include <cmath>

#define SQRT_2 (1.4142135623)

Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    // TODO: set these more appropriately
    const unsigned int threadsPerBlock = 64;
    const unsigned int blocks = (int) ceil((float) pts.size / threadsPerBlock);
    Clustering clusters(pts.size);
    //const float EPS_SQ = epsilon * epsilon;
    float *dev_coords;
    CUDA_CALL(cudaMalloc((void**)&dev_coords, pts.size * 2 * sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_coords, pts.data, pts.size * 2 * sizeof(float),
                         cudaMemcpyHostToDevice));
    BBox bbox = cuda_extent(pts, dev_coords, blocks, threadsPerBlock);

    const float side_len = epsilon / SQRT_2;
    int grid_x_size = (int) ((bbox.max_x - bbox.min_x) / side_len) + 1;
    int grid_y_size = (int) ((bbox.max_y - bbox.min_y) / side_len) + 1;

    // label each point with a grid bin
    uint *dev_pt_ids, *dev_grid_labels;
    CUDA_CALL(cudaMalloc((void**)&dev_pt_ids, pts.size * sizeof(uint)));
    CUDA_CALL(cudaMalloc((void**)&dev_grid_labels, pts.size * sizeof(uint)));
    callGridLabelKernel(blocks, threadsPerBlock, dev_pt_ids, dev_grid_labels,
                        dev_coords, bbox.min_x, bbox.min_y, side_len,
                        grid_x_size, pts.size);

    // Insert into hash table
    CUDPPHandle *cudpp;
    CUDPP_CALL(cudppCreate(cudpp));
    CUDPPHashTableConfig hashconf = {
        CUDPP_MULTIVALUE_HASH_TABLE,
        (uint) pts.size,
        1.25 // extra memory factor (1.05 to 2.0, trades memory for build speed)
    };
    CUDPPHandle *grid;
    CUDPP_CALL(cudppHashTable(*cudpp, grid, &hashconf));
    CUDPP_CALL(cudppHashInsert(*grid, dev_grid_labels, dev_pt_ids, pts.size));
    unsigned int **d_values;
    CUDPP_CALL(cudppMultivalueHashGetAllValues(*grid, d_values));
    unsigned int **d_index_counts;
    CUDPP_CALL(cudppMultivalueHashGetIndexCounts(*grid, d_index_counts));
    unsigned int unique_key_count;
    CUDPP_CALL(cudppMultivalueHashGetUniqueKeyCount(*grid, &unique_key_count));
    bool *isCore;
    CUDA_CALL(cudaMalloc((void**)&isCore, pts.size * sizeof(bool)));
    CUDA_CALL(cudaMemset(isCore, 0, pts.size * sizeof(bool)));
    callGridMarkCoreCells(blocks, threadsPerBlock, d_index_counts,
                          unique_key_count, d_values, isCore, min_points);
    
    CUDPP_CALL(cudppDestroyHashTable(*cudpp, *grid));
    
    CUDPP_CALL(cudppDestroy(*cudpp));
    CUDA_CALL(cudaFree(dev_coords));
    return clusters;
}

BBox cuda_extent(PointSet &pts, float *dev_coords,
                 const unsigned int blocks,
                 const unsigned int threadsPerBlock) {
    BBox bbox = {.min_x = 1000, .min_y = 1000, .max_x = -1000, .max_y = -1000 };
    float *dev_max_x, *dev_max_y, *dev_min_x, *dev_min_y;
    CUDA_CALL(cudaMalloc((void**)&dev_max_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_max_y, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_min_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_min_y, sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_max_x, &bbox.max_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_max_y, &bbox.max_y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_min_x, &bbox.min_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_min_y, &bbox.min_y, sizeof(float), cudaMemcpyHostToDevice));
    cudaCallMaxXYKernel(blocks, threadsPerBlock, dev_coords, pts.size * 2,
                        dev_max_x, dev_max_y);
    cudaCallMinXYKernel(blocks, threadsPerBlock, dev_coords, pts.size * 2,
                        dev_min_x, dev_min_y);
    CUDA_CALL(cudaMemcpy(&bbox.max_x, dev_max_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&bbox.max_y, dev_max_y, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&bbox.min_x, dev_min_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&bbox.min_y, dev_min_y, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(dev_max_x));
    CUDA_CALL(cudaFree(dev_max_y));
    CUDA_CALL(cudaFree(dev_min_x));
    CUDA_CALL(cudaFree(dev_min_y));
    return bbox;
}
