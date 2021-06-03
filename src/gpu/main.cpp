#include "point_set.hpp"
#include "clustering.hpp"
#include "load_taxi.hpp"
#include "cuda_utils.hpp"
#include "delaunay.cuh"
#include "cmdline_parse.hpp"

int main(int argc, char **argv) {
    int n_pts, min_points;
    float epsilon;
    bool print;
    parse(argc, argv, &n_pts, &min_points, &epsilon, &print);
    PointSet pts = get_n_pickups(n_pts, nullptr);
    auto start_time = high_resolution_clock::now();
    Clustering c = delaunay_dbscan(pts, epsilon, min_points);
    auto end_time = high_resolution_clock::now();
    if (print) {
        c.print();
    }
    else {
        duration<double, std::milli> ms_elapsed = end_time - start_time;
        cout << ms_elapsed.count() << "ms\n";
    }
    return 0;
}
