#include "point_set.hpp"
#include "clustering.hpp"
#include "load_taxi.hpp"
#include "cuda_utils.h"
#include "delaunay_gpu.cuh"

#include <iostream>

using namespace std;

int main(void) {
    PointSet ps = get_n_pickups(1024, nullptr);
    BBox bbox = pts.extent();
    float max_x = -1000, max_y = -1000; // definitely less than gps coords
    float *dev_max_x, *dev_max_y;
    CUDA_CALL(cudaMalloc((void**)dev_max_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)dev_max_y, sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_max_x, &max_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_max_y, &max_y, sizeof(float), cudaMemcpyHostToDevice));
    cudaCallMaxXYKernel(1024 / 32, 32, ps.data, ps.size * 2, dev_max_x, dev_max_y);
    CUDA_KERNEL_CHECK();
    CUDA_CALL(cudaMemcpy(&max_x, dev_max_x, sizeof(float), cudaMemcpyDeviceToHost));
    cout << "CPU Result" << bbox.max_x << bbox.max_y << "\n";
    cout << "GPU Result" << max_x << max_y << "\n";
    return 0;
}
