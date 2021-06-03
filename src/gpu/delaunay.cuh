#include "clustering.hpp"
#include "point_set.hpp"

#include <vector>

Clustering *delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points);

BBox cuda_extent(PointSet &pts, float *dev_coords,
                 const unsigned int blocks,
                 const unsigned int threadsPerBlock);

std::vector<int> neighbor_cell_ids(int x, int y, int grid_x_size, int grid_y_size);
