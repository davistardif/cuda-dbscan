#pragma once
#include "point_set.hpp"
#include "clustering.hpp"

Clustering naive_dbscan(PointSet &pts, float epsilon, unsigned int min_points);
