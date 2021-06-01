#include "clustering.hpp"
#include "point_set.hpp"

Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points);

BBox cuda_extent(PointSet &pts, float *dev_coords,
                 const unsigned int blocks,
                 const unsigned int threadsPerBlock);
