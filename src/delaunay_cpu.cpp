#include "clustering.hpp"
#include "point_set.hpp"

Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    Clustering clusters(pts.size);
    BBox bbox = pts.extent();


    return clusters;
}
