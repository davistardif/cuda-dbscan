#include "point_set.hpp"
#include "clustering.hpp"
#include "load_taxi.hpp"
#include "cuda_utils.hpp"
#include "delaunay.cuh"

int main(void) {
    PointSet pts = get_n_pickups(1024, nullptr);
    delaunay_dbscan(pts, 0.1, 100);
    return 0;
}
