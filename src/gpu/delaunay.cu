#include "delaunay.cuh"

#include "clustering.hpp"
#include "point_set.hpp"
#include "cuda_utils.hpp"
#include "minmax.cuh"
#include "grid.cuh"

#include "cudpp.h"
#include "cudpp_hash.h"
#include "cudpp_config.h"

#include "GpuDelaunay.h"

#include <cmath>
#include <cassert>
#include <vector>

#define SQRT_2 (1.4142135623)

using std::vector;

Clustering *delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    // TODO: set these more appropriately
    const unsigned int threadsPerBlock = 64;
    const unsigned int blocks = (int) ceil((float) pts.size / threadsPerBlock);
    Clustering *clusters = new Clustering(pts.size);
    const float EPS_SQ = epsilon * epsilon;
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
    CUDPPHandle cudpp;
    CUDPP_CALL(cudppCreate(&cudpp));
    CUDPPHashTableConfig *hashconf = new CUDPPHashTableConfig;
    *hashconf = {
        CUDPP_MULTIVALUE_HASH_TABLE,
        (uint) pts.size,
        1.25 // extra memory factor (1.05 to 2.0, trades memory for build speed)
    };
    CUDPPHandle grid;
    CUDPP_CALL(cudppHashTable(cudpp, &grid, hashconf));
    CUDPP_CALL(cudppHashInsert(grid, dev_grid_labels, dev_pt_ids, pts.size));

    // Mark core points where cell has >= min points
    unsigned int **d_values;
    CUDPP_CALL(cudppMultivalueHashGetAllValues(grid, d_values));
    unsigned int **d_index_counts;
    CUDPP_CALL(cudppMultivalueHashGetIndexCounts(grid, d_index_counts));
    unsigned int unique_key_count;
    CUDPP_CALL(cudppMultivalueHashGetUniqueKeyCount(grid, &unique_key_count));
    bool *d_isCore, *isCore;
    CUDA_CALL(cudaMalloc((void**)&d_isCore, pts.size * sizeof(bool)));
    CUDA_CALL(cudaMemset(d_isCore, 0, pts.size * sizeof(bool)));
    callGridMarkCoreCells(blocks, threadsPerBlock, *d_index_counts,
                          unique_key_count, *d_values, d_isCore, min_points);
    
    // Find remaining core points
    isCore = (bool *) malloc(pts.size * sizeof(bool));
    assert(isCore != nullptr);
    CUDA_CALL(cudaMemcpy(isCore, d_isCore, pts.size * sizeof(bool), cudaMemcpyDeviceToHost));
    uint *d_query_keys, *d_results;
    CUDA_CALL(cudaMalloc((void**)&d_query_keys, 21 * sizeof(uint)));
    CUDA_CALL(cudaMalloc((void**)&d_results, 21 * sizeof(uint2)));
    for (int i = 0; i < pts.size; i++) {
        if (!isCore[i]) {
            int x = (int) ((pts.get_x(i) - bbox.min_x) / side_len);
            int y = (int) ((pts.get_y(i) - bbox.min_y) / side_len);
            vector<int> ncells = neighbor_cell_ids(x, y, grid_x_size, grid_y_size);
            CUDA_CALL(cudaMemcpy(d_query_keys, (uint *) ncells.data(),
                                 ncells.size() * sizeof(uint), cudaMemcpyHostToDevice));
            CUDPP_CALL(cudppHashRetrieve(grid, d_query_keys, d_results, ncells.size()));
            callGridCheckCore(dev_coords, d_results, ncells.size(), *d_values,
                              d_isCore, min_points, EPS_SQ,
                              pts.get_x(i), pts.get_y(i), i);
        }
    }
    CUDA_CALL(cudaMemcpy(isCore, d_isCore, pts.size * sizeof(bool), cudaMemcpyDeviceToHost));
    // Delaunay triangulation of core points
    Point2HVec core_pts;
    vector<int> core_idx;
    Point2 p;
    for (int i = 0; i < pts.size; i++) {
        if (isCore[i]) {
            p._p[0] = pts.get_x(i);
            p._p[1] = pts.get_y(i);
            core_pts.push_back(p);
            core_idx.push_back(i);
        }
    }
    GpuDel gdel;
    GDel2DInput gdelIn;
    gdelIn.pointVec = core_pts;
    GDel2DOutput gdelOut;
    gdel.compute(gdelIn, &gdelOut);
              
    CUDPP_CALL(cudppDestroyHashTable(cudpp, grid));
    
    CUDPP_CALL(cudppDestroy(cudpp));
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

vector<int> neighbor_cell_ids(int x, int y, int grid_x_size, int grid_y_size) {
    /* Returns a vector of cell id's in grid which are epsilon neighbors
       of (x,y) and not out of bounds.
       includes (x,y) itself
    */
    int cell_r_c[] = {
        (y-2), x - 1,
        (y-2), x,
        (y-2), x + 1,
        (y-1), x - 2,
        (y-1), x - 1,
        (y-1), x,
        (y-1), x + 1,
        (y-1), x + 2,
        y, x - 2,
        y, x - 1,
        y, x + 1,
        y, x + 2,
        (y+2), x - 1,
        (y+2), x,
        (y+2), x + 1,
        (y+1), x - 2,
        (y+1), x - 1,
        (y+1), x,
        (y+1), x + 1,
        (y+1), x + 2
    };
    vector<int> valid_cells;
    valid_cells.reserve(21);
    valid_cells.push_back(y * grid_x_size + x);
    for (int i = 0; i < 20; i++) { // 20 neighbor cell possibilities
        int r = cell_r_c[i*2];
        int c = cell_r_c[i*2 + 1];
        if (c < 0 || r < 0 || c >= grid_x_size || r >= grid_y_size) {
            // out of bounds
            continue;
        }
        int id = r * grid_x_size + c;
        valid_cells.push_back(id);
    }
    return valid_cells;
}
