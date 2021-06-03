#include "point_set.hpp"
#include "clustering.hpp"
#include "load_taxi.hpp"
#include "cuda_utils.hpp"
#include "delaunay.cuh"
#include "cmdline_parse.hpp"

int main(int argc, char **argv) {
    PointSet pts = get_n_pickups(1024, nullptr);
    Clustering c = delaunay_dbscan(pts, 0.1, 100);
    c.print();
    return 0;
}
