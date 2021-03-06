#include "clustering.hpp"
#include "point_set.hpp"
#include "cpu_dbscan.hpp"
#include "disjoint_set.hpp"

#include <delaunator.hpp>

#include <list>
#include <vector>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <cstring>


#define SQRT_2 (1.4142135623)

using namespace std;

vector<int> neighbor_cell_ids(unordered_map<int, vector<int>> &grid,
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

Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    Clustering clusters(pts.size);
    const float EPS_SQ = epsilon * epsilon;
    // Grid computation
    /* Parallelization idea:
       simple CUDA reduction kernel to find the min and max x,y coordinates
    */
    BBox bbox = pts.extent();
    float side_len = epsilon / SQRT_2;
    int grid_x_size = (int) ((bbox.max_x - bbox.min_x) / side_len) + 1;
    int grid_y_size = (int) ((bbox.max_y - bbox.min_y) / side_len) + 1;
    /* Parallelization idea:
       Simple CUDA kernel can compute the grid index for each point,
       and can use an external library for CUDA compatible hash table
       e.g. CUDPP: http://cudpp.github.io/cudpp/2.2/hash_overview.html
    */
    unordered_map<int, vector<int>> grid;
    for (int i = 0; i < pts.size; i++) {
        int x = (int) ((pts.get_x(i) - bbox.min_x) / side_len);
        int y = (int) ((pts.get_y(i) - bbox.min_y) / side_len);
        int idx = y*grid_x_size + x;
        auto it = grid.find(idx);
        if (it == grid.end()) {
            grid[idx] = vector<int>();
            grid[idx].push_back(i);
        }
        else {
            it->second.push_back(i);
        }
    }
    
    // Mark core
    /*
      Parallelization idea:
      CUDA kernel will work on keys of grid hash table in parallel
      Perhaps two stages, one for cells with >= min_points and one 
      for searching neighboring cells to avoid warp divergence
    */
    int num_core = 0, num_core_cells = 0;
    bool *is_core = (bool*) malloc(pts.size * sizeof(bool));
    assert(is_core != nullptr);
    memset(is_core, 0, pts.size * sizeof(bool));
    unordered_map<int, int> core_cell_idx;
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
        int x = it->first % grid_x_size;
        int y = it->first / grid_x_size;
        bool cell_has_core = false;
        for (int i = 0; i < it->second.size(); i++) {
            // scan over neighboring cells (anything else is too far away to matter)
            int pt1 = it->second[i];
            vector<int> nbrs = neighbor_cell_ids(grid, x, y, grid_x_size, grid_y_size);
            // automatically include all points in the same grid cell
            int nbr_count = it->second.size();
            int j = 0;
            while (nbr_count < min_points && j < nbrs.size()) {
                vector<int> to_check = grid[nbrs[j]];
                for (int &pt2 : to_check) {
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
    //printf("Number of core points: %d\n", num_core);

    // Delaunay Triangulation of all core points
    /*
      Parallelization idea:
      Use external library (gDel2d)
    */
    vector<double> core_pts;
    vector<int> core_idx;
    core_pts.reserve(num_core);
    core_idx.reserve(num_core);
    for (int i = 0; i < pts.size; i++) {
        if (is_core[i]) {
            core_pts.push_back(pts.get_x(i));
            core_pts.push_back(pts.get_y(i));
            core_idx.push_back(i);
        }
    }
    delaunator::Delaunator delaunay(core_pts);

    // Keep only edges which cross cells and have length <= epsilon
    /** 
        Parallelization idea:
        Implement union-find data structure for cuda
        following this paper: 
        https://userweb.cs.txstate.edu/~mb92/papers/hpdc18.pdf
        Reference code is also available at:
        https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/
    */
    DisjointSet dj_set(num_core_cells);
    for (int i = 0; i < delaunay.triangles.size(); i++) {
        int pt1 = core_idx[delaunay.triangles[i]];
        int pt2 = core_idx[delaunay.triangles[(i % 3 == 2) ? i - 2 : i + 1]];
        //printf("edge: (%d, %d)\n", pt1, pt2);
        // note: some edges are duplicated
        if (pts.dist_sq(pt1, pt2) <= EPS_SQ) {
            int x1 = (int) ((pts.get_x(pt1) - bbox.min_x) / side_len);
            int y1 = (int) ((pts.get_y(pt1) - bbox.min_y) / side_len);
            int x2 = (int) ((pts.get_x(pt2) - bbox.min_x) / side_len);
            int y2 = (int) ((pts.get_y(pt2) - bbox.min_y) / side_len);
            if (x1 != x2 || y1 != y2) {
                // Keep edge
                //printf("Keeping edge (%d, %d) between cells (%d, %d) (%d, %d)\n", pt1, pt2, x1, y1, x2, y2);
                int idx1 = core_cell_idx[y1*grid_x_size + x1];
                int idx2 = core_cell_idx[y2*grid_x_size + x2];
                dj_set.union_sets(idx1, idx2);
            }
        }
    }
    // set cluster id based on disjoint set representatives (i.e. connected components)
    for (auto it = core_cell_idx.begin(); it != core_cell_idx.end(); it++) {
        int cluster_id = dj_set.find_set(it->second) + 1;
        for (auto pt_it = grid[it->first].begin(); pt_it != grid[it->first].end(); ++pt_it) {
            if (is_core[*pt_it])
                clusters.set_cluster(*pt_it, cluster_id);
        }
    }
    // Assign border points
    /**
       Parallelization idea:
       CUDA kernel which will be very similar to mark core
    */
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
            vector<int> &nbr_pts = grid[nbrs[j]];
            for (int &nbr_pt : nbr_pts) {
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
    return clusters;
}
