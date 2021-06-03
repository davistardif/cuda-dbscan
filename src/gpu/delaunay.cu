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
#include <unordered_map>

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
    uint *grid_labels = (uint *) malloc(pts.size * sizeof(uint));
    CUDA_CALL(cudaMemcpy(grid_labels, dev_grid_labels, pts.size * sizeof(uint), cudaMemcpyDeviceToHost));
    // Insert into hash table
    unordered_map<uint, vector<uint>> grid;
    for (uint i = 0; i < pts.size; i++) {
      uint idx = grid_labels[i];
      auto it = grid.find(idx);
        if (it == grid.end()) {
            grid[idx] = vector<uint>();
            grid[idx].push_back(i);
        }
        else {
            it->second.push_back(i);
        }
    }
    
    // Mark core
    int num_core = 0, num_core_cells = 0;
    bool *is_core = (bool*) malloc(pts.size * sizeof(bool));
    assert(is_core != nullptr);
    memset(is_core, 0, pts.size * sizeof(bool));
    unordered_map<uint, uint> core_cell_idx;
    for (auto it = grid.begin(); it != grid.end(); it++) {
        if (it->second.size() >= min_points) {
            // mark all points in the cell as core points
            core_cell_idx[it->first] = num_core_cells++;
            for (int i = 0; i < it->second.size(); i++) {
                is_core[it->second[i]] = true;
            }
            num_core += it->second.size();
            continue;
        }
        // otherwise, need to check each point individually
        uint x = it->first % grid_x_size;
        uint y = it->first / grid_x_size;
        bool cell_has_core = false;
        for (uint i = 0; i < it->second.size(); i++) {
            // scan over neighboring cells (anything else is too far away to matter)
            uint pt1 = it->second[i];
            vector<int> nbrs = neighbor_cell_ids(grid, x, y, grid_x_size, grid_y_size);
            // automatically include all points in the same grid cell
            int nbr_count = it->second.size();
            int j = 0;
            while (nbr_count < min_points && j < nbrs.size()) {
                vector<uint> to_check = grid[nbrs[j]];
                for (uint &pt2 : to_check) {
                    if (pts.dist_sq(pt1, pt2) <= EPS_SQ)
                        nbr_count++;
                }
                j++;
            }
            if (nbr_count >= min_points) {
                is_core[pt1] = true;
                num_core++;
                if (!cell_has_core) {
                    core_cell_idx[it->first] = num_core_cells++;
                    cell_has_core = true;
                }
            }
        }
    }
    // Delaunay triangulation of core points
    Point2HVec core_pts;
    vector<uint> core_idx;
    core_pts.reserve(num_core);
    core_idx.reserve(num_core);
    Point2 p;
    for (uint i = 0; i < pts.size; i++) {
        if (is_core[i]) {
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
              
    //CUDPP_CALL(cudppDestroyHashTable(cudpp, grid));
    
    //CUDPP_CALL(cudppDestroy(cudpp));
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

/*
vector<int> neighbor_cell_ids(int x, int y, int grid_x_size, int grid_y_size) {
    /* Returns a vector of cell id's in grid which are epsilon neighbors
       of (x,y) and not out of bounds.
       includes (x,y) itself
    * /
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
*/
vector<int> neighbor_cell_ids(unordered_map<uint, vector<uint>> &grid,
                              int x, int y, int grid_x_size, int grid_y_size) {
    /* Returns a vector of cell id's in grid which are epsilon neighbors
       of (x,y) and are non-empty
       Note: does not include (x,y) itself
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
    for (int i = 0; i < 20; i++) { // 20 neighbor cell possibilities
        int r = cell_r_c[i*2];
        int c = cell_r_c[i*2 + 1];
        if (c < 0 || r < 0 || c >= grid_x_size || r >= grid_y_size) {
            // out of bounds
            continue;
        }
        int id = r * grid_x_size + c;
        if (grid.count(id) > 0) {
            // only return nonempty cells
            valid_cells.push_back(id);
        }
    }
    return valid_cells;
}
