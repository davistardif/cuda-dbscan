#include <assert.h>

#include "point_set.hpp"
#include "clustering.hpp"
#include "cpu_dbscan.hpp"
#include "load_taxi.hpp"
#include "cmdline_parse.hpp"

#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

void test_dbscan(void) {
    // Create a test dataset of 10 points to ensure alg. works correctly
    PointSet pts(10);
    // Clusters: (0,1,2,3) (4,5,6,7) 3, 7 are border; 8, 9 are noise
    // eps = 2, min_points = 3
    pts.set(0, 0, 0);
    pts.set(1, -1, 0);
    pts.set(2, 1, 0);
    pts.set(3, 3, 0);
    pts.set(4, 0, 100);
    pts.set(5, -1, 100);
    pts.set(6, 1, 100);
    pts.set(7, 3, 100);
    pts.set(8, -5, -5);
    pts.set(9, -100, 200);
    Clustering c = naive_dbscan(pts, 2, 3);
    int clu1 = c.get_cluster(0);
    int clu2 = c.get_cluster(4);
    for (int i = 0; i < 10; i++) {
        assert(c.is_labeled(i));
        if (i < 4) {
            assert(c.get_cluster(i) == clu1);
        }
        else if (i < 8) {
            assert(c.get_cluster(i) == clu2);
        }
        else {
            assert(c.is_noise(i));
        }
        if (i == 3 || i == 7) {
            assert(c.is_border(i));
        }
    }
}

void test_delaunay_dbscan(void) {
    // Create a test dataset of 10 points to ensure alg. works correctly
    PointSet pts(10);
    // Clusters: (0,1,2,3) (4,5,6,7) 3, 7 are border; 8, 9 are noise
    // eps = 2, min_points = 3
    pts.set(0, 0, 0);
    pts.set(1, -1, 0);
    pts.set(2, 1, 0);
    pts.set(3, 3, 0);
    pts.set(4, 0, 100);
    pts.set(5, -1, 100);
    pts.set(6, 1, 100);
    pts.set(7, 3, 100);
    pts.set(8, -5, -5);
    pts.set(9, -100, 200);
    
    Clustering c = delaunay_dbscan(pts, 2, 3);

    int clu1 = c.get_cluster(0);
    int clu2 = c.get_cluster(4);
    for (int i = 0; i < 10; i++) {
        assert(c.is_labeled(i));
        if (i < 4) {
            assert(c.get_cluster(i) == clu1);
        }
        else if (i < 8) {
            assert(c.get_cluster(i) == clu2);
        }
        else {
            assert(c.is_noise(i));
        }
        if (i == 3 || i == 7) {
            assert(c.is_border(i));
        }
    }
}

int main(void) {
    /*
      Code for comparing naive and delaunay cpu dbscan
   
    test_dbscan();
    test_delaunay_dbscan();
    PointSet ps = get_n_pickups(10000, nullptr);
    cout << "Naive CPU DBSCAN...\n";
    auto start_time = high_resolution_clock::now();
    Clustering c1 = naive_dbscan(ps, .004, 30);
    auto end_time = high_resolution_clock::now();
    duration<double, std::milli> ms_elapsed = end_time - start_time;
    cout << ms_elapsed.count() << "ms\n";
    cout << "Delaunay CPU DBSCAN...\n";
    start_time = high_resolution_clock::now();
    Clustering c2 = delaunay_dbscan(ps, .004, 30);
    end_time = high_resolution_clock::now();
    ms_elapsed = end_time - start_time;
    cout << ms_elapsed.count() << "ms\n";
    */
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
