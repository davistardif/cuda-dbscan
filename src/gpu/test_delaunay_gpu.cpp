#include "point_set.hpp"
#include "clustering.hpp"
#include "load_taxi.hpp"
#include "cuda_utils.hpp"
#include "minmax.cuh"

#include <iostream>
#include <cstdio>

using namespace std;

int test_minmax(void) {
    PointSet pts = get_n_pickups(1024, nullptr);
    BBox bbox = pts.extent();
    float max_x = -1000, max_y = -1000; // definitely less than gps coords
    float *dev_max_x, *dev_max_y;
    float min_x = 1000, min_y = 1000;
    float *dev_min_x, *dev_min_y;
    float *dev_coords;
    CUDA_CALL(cudaMalloc((void**)&dev_max_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_max_y, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_min_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_min_y, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_coords, pts.size * 2 * sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_max_x, &max_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_max_y, &max_y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_min_x, &min_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_min_y, &min_y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_coords, pts.data, pts.size * 2 * sizeof(float),
                  cudaMemcpyHostToDevice));
    cudaCallMaxXYKernel(1024 / 32, 32, dev_coords, pts.size * 2,
                        dev_max_x, dev_max_y);
    CUDA_KERNEL_CHECK();
    cudaCallMinXYKernel(1024 / 32, 32, dev_coords, pts.size * 2,
                        dev_min_x, dev_min_y);
    CUDA_KERNEL_CHECK();
    CUDA_CALL(cudaMemcpy(&max_x, dev_max_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&max_y, dev_max_y, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&min_x, dev_min_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&min_y, dev_min_y, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(dev_max_x));
    CUDA_CALL(cudaFree(dev_max_y));
    CUDA_CALL(cudaFree(dev_min_x));
    CUDA_CALL(cudaFree(dev_min_y));
    CUDA_CALL(cudaFree(dev_coords));
    printf("CPU Result: (%f, %f) - (%f, %f)\n", bbox.min_x, bbox.min_y,
           bbox.max_x, bbox.max_y);
    printf("GPU Result: (%f, %f) - (%f, %f)\n", min_x, min_y, max_x, max_y);
    return 0;
}
