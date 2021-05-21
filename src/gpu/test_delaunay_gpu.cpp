#include "point_set.hpp"
#include "clustering.hpp"
#include "load_taxi.hpp"
#include "cuda_utils.hpp"
#include "delaunay.cuh"

#include <iostream>

using namespace std;

int main(void) {
    PointSet pts = get_n_pickups(1024, nullptr);
    BBox bbox = pts.extent();
    float max_x = -1000, max_y = -1000; // definitely less than gps coords
    float *dev_max_x, *dev_max_y;
    float *dev_coords;
    CUDA_CALL(cudaMalloc((void**)&dev_max_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_max_y, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_coords, pts.size * 2 * sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_max_x, &max_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_max_y, &max_y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_coords, pts.data, pts.size * 2 * sizeof(float),
                  cudaMemcpyHostToDevice));
    cudaCallMaxXYKernel(1024 / 32, 32, dev_coords, pts.size * 2,
                        dev_max_x, dev_max_y);
    CUDA_KERNEL_CHECK();
    CUDA_CALL(cudaMemcpy(&max_x, dev_max_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&max_y, dev_max_y, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(dev_max_x));
    CUDA_CALL(cudaFree(dev_max_y));
    CUDA_CALL(cudaFree(dev_coords));
    cout << "CPU Result" << bbox.max_x << bbox.max_y << "\n";
    cout << "GPU Result" << max_x << max_y << "\n";
    return 0;
}
