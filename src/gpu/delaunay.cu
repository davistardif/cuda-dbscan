#include "delaunay.cuh"

#include "clustering.hpp"
#include "point_set.hpp"
#include "cuda_utils.hpp"
#include "minmax.cuh"
#include "grid.cuh"
#include "disjoint_set.hpp"

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

Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    // TODO: set these more appropriately
    const unsigned int threadsPerBlock = 64;
    const unsigned int blocks = (int) ceil((float) pts.size / threadsPerBlock);
    Clustering clusters(pts.size);
    const float EPS_SQ = epsilon * epsilon;
    float *dev_coords;
    CUDA_CALL(cudaMalloc((void**)&dev_coords, pts.size * 2 * sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_coords, pts.data, pts.size * 2 * sizeof(float),
                         cudaMemcpyHostToDevice));
    BBox bbox = cuda_extent(pts, dev_coords, blocks, threadsPerBlock);

    const float side_len = epsilon / SQRT_2;
    int grid_x_size = (int) ((bbox.max_x - bbox.min_x) / side_len) + 1;
    int grid_y_size = (int) ((bbox.max_y - bbox.min_y) / side_len) + 1;
    /*
      Unfortunately I can't get the cudpp hashtable to work with
      gDel2D. I think that maybe they require different versions of 
      thrust/CUB but I don't have any more time to mess with it. Anyways,
      here is the code that I was using to create the hash table and find 
      core points on the GPU:
      
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
    unsigned int *d_values;
    CUDA_CALL(cudaMalloc((void**)&d_values, pts.size * sizeof(unsigned int)));
    CUDPP_CALL(cudppMultivalueHashGetAllValues(grid, &d_values));
    unsigned int unique_key_count;
    CUDPP_CALL(cudppMultivalueHashGetUniqueKeyCount(grid, &unique_key_count));
    unsigned int *d_index_counts;
    CUDA_CALL(cudaMalloc((void**)&d_index_counts, unique_key_count * 2 * sizeof(unsigned int)));
    CUDPP_CALL(cudppMultivalueHashGetIndexCounts(grid, &d_index_counts));
    
    bool *d_isCore, *isCore;
    CUDA_CALL(cudaMalloc((void**)&d_isCore, pts.size * sizeof(bool)));
    CUDA_CALL(cudaMemset(d_isCore, 0, pts.size * sizeof(bool)));
    callGridMarkCoreCells(blocks, threadsPerBlock, d_index_counts,
                          unique_key_count, d_values, d_isCore, min_points);
    
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
            callGridCheckCore(dev_coords, d_results, ncells.size(), d_values,
                              d_isCore, min_points, EPS_SQ,
                              pts.get_x(i), pts.get_y(i), i);
        }
    }
    CUDA_CALL(cudaMemcpy(isCore, d_isCore, pts.size * sizeof(bool), cudaMemcpyDeviceToHost));
    */
    /* Below is code for finding core points with CPU, as is done
       in the CPU demo. 
    */
    // label each point with a grid bin
    uint *dev_pt_ids, *dev_grid_labels;
    CUDA_CALL(cudaMalloc((void**)&dev_pt_ids, pts.size * sizeof(uint)));
    CUDA_CALL(cudaMalloc((void**)&dev_grid_labels, pts.size * sizeof(uint)));
    callGridLabelKernel(blocks, threadsPerBlock, dev_pt_ids, dev_grid_labels,
                        dev_coords, bbox.min_x, bbox.min_y, side_len,
                        grid_x_size, pts.size);
    CUDA_CALL(cudaFree(dev_pt_ids)); // not needed for CPU version
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
    /*
    int *component_id = (int *) malloc(num_core_cells * sizeof(int));
    // nindex[v] stores beginning index of neighbor list for vertex v in nlist
    // last element of nindex stores length of nlist
    int *nindex = (int *) malloc((num_core_cells + 1) * sizeof(int));
    vector<int> nlist;
    */
    DisjointSet dj_set(num_core_cells);
    for (Tri &tri : gdelOut.triVec) {
        int pt1 = core_idx[tri._v[0]];
        int pt2 = core_idx[tri._v[1]];
        int pt3 = core_idx[tri._v[2]];
        int x1 = (int) ((pts.get_x(pt1) - bbox.min_x) / side_len);
        int y1 = (int) ((pts.get_y(pt1) - bbox.min_y) / side_len);
        int x2 = (int) ((pts.get_x(pt2) - bbox.min_x) / side_len);
        int y2 = (int) ((pts.get_y(pt2) - bbox.min_y) / side_len);
        int x3 = (int) ((pts.get_x(pt3) - bbox.min_x) / side_len);
        int y3 = (int) ((pts.get_y(pt3) - bbox.min_y) / side_len);
        if (pts.dist_sq(pt1, pt2) <= EPS_SQ && (x1 != x2 || y1 != y2)) {
            int idx1 = core_cell_idx[y1*grid_x_size + x1];
            int idx2 = core_cell_idx[y2*grid_x_size + x2];
            dj_set.union_sets(idx1, idx2);
        }
        if (pts.dist_sq(pt1, pt3) <= EPS_SQ && (x1 != x3 || y1 != y3)) {
            int idx1 = core_cell_idx[y1*grid_x_size + x1];
            int idx2 = core_cell_idx[y3*grid_x_size + x3];
            dj_set.union_sets(idx1, idx2);
        }
        if (pts.dist_sq(pt2, pt3) <= EPS_SQ && (x2 != x3 || y2 != y3)) {
            int idx1 = core_cell_idx[y2*grid_x_size + x2];
            int idx2 = core_cell_idx[y3*grid_x_size + x3];
            dj_set.union_sets(idx1, idx2);
        }
    }
    // set cluster id based on disjoint set representatives
    for (auto it = core_cell_idx.begin(); it != core_cell_idx.end(); it++) {
        int cluster_id = dj_set.find_set(it->second) + 1;
        for (auto pt_it = grid[it->first].begin(); pt_it != grid[it->first].end(); ++pt_it) {
            if (is_core[*pt_it])
                clusters.set_cluster(*pt_it, cluster_id);
        }
    }
    // Assign border points
    // Unfortunately this also cannot run in cuda without hash table support
    for (int i = 0; i < pts.size; i++) {
        if (clusters.is_labeled(i))
            continue;
        int x = (int) ((pts.get_x(i) - bbox.min_x) / side_len);
        int y = (int) ((pts.get_y(i) - bbox.min_y) / side_len);
        vector<int> nbrs = neighbor_cell_ids(grid, x, y, grid_x_size, grid_y_size);
        bool done = false;
        int j = 0;
        while (!done && j < nbrs.size()) {
            if (core_cell_idx.count(nbrs[j]) == 0) {
                j++;
                continue; // has no core points, skip this cell
            }
            vector<uint> &nbr_pts = grid[nbrs[j]];
            for (uint &nbr_pt : nbr_pts) {
                if (clusters.is_core(nbr_pt) && pts.dist_sq(i, nbr_pt) <= EPS_SQ) {
                    clusters.set_border(i, clusters.get_cluster(nbr_pt));
                    done = true;
                    break;
                }
            }
            j++;
        }
        if (!done) {
            clusters.set_noise(i);
        }
    }
    free(is_core);
    
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

/* This was a modified version of neighbor_cell_ids from CPU demo
   that was adapted for use with the CUDPP hash table
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
    
