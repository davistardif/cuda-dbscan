#include "clustering.hpp"
#include "point_set.hpp"
#include "cpu_dbscan.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include <unordered_map>

#define SQRT_2 (1.4142135623)

using namespace std;

Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    Clustering clusters(pts.size);
    BBox bbox = pts.extent();
    float side_len = epsilon / SQRT_2;
    int grid_x_size = (int) ((bbox.max_x - bbox.min_x) / side_len) + 1;
    int grid_y_size = (int) ((bbox.max_y - bbox.min_y) / side_len) + 1;
    unordered_map<long, vector<int>> grid;
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
    // Testing: Print out the grid
    auto it = grid.begin();
    while (it != grid.end()) {
        int x = it->first % grid_x_size;
        int y = it->first / grid_x_size;
        printf("(%d,%d): ", x, y);
        for (int i = 0; i < it->second.size(); i++) {
            printf("%d,", it->second[i]);
        }
        printf("\n");
        it++;
    }
    return clusters;
}
