#pragma once
#include "point_set.hpp"
#include "clustering.hpp"

Clustering naive_dbscan(PointSet &pts, float epsilon, unsigned int min_points);
Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points);
