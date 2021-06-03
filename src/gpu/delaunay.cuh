#include "clustering.hpp"
#include "point_set.hpp"

#include <vector>
#include <unordered_map>

using namespace std;

Clustering *delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points);

BBox cuda_extent(PointSet &pts, float *dev_coords,
                 const unsigned int blocks,
                 const unsigned int threadsPerBlock);

//std::vector<int> neighbor_cell_ids(int x, int y, int grid_x_size, int grid_y_size);
vector<int> neighbor_cell_ids(unordered_map<uint, vector<uint>> &grid,
                              int x, int y, int grid_x_size, int grid_y_size);