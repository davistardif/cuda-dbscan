#include "delaunay.cuh"

#include "clustering.hpp"
#include "point_set.hpp"
#include "cuda_utils.hpp"
#include "minmax.cuh"

Clustering delaunay_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    Clustering clusters(pts.size);
    //const float EPS_SQ = epsilon * epsilon;
    float *dev_coords;
    CUDA_CALL(cudaMalloc((void**)&dev_coords, pts.size * 2 * sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_coords, pts.data, pts.size * 2 * sizeof(float),
                         cudaMemcpyHostToDevice));
    BBox bbox = cuda_extent(pts, dev_coords);


    CUDA_CALL(cudaFree(dev_coords));
    return clusters;
}

BBox cuda_extent(PointSet &pts, float *dev_coords,
                 const unsigned int blocks,
                 const unsigned int threadsPerBlock) {
    BBox bbox = {.min_x = 1000, .min_y = 1000, .max_x = -1000, .max_y = -1000 };
    float *dev_max_x, *dev_max_y, *dev_min_x, *dev_min_y;
    CUDA_CALL(cudaMalloc((void**)&dev_max_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_max_y, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_min_x, sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_min_y, sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_max_x, &bbox.max_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_max_y, &bbox.max_y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_min_x, &bbox.min_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_min_y, &bbox.min_y, sizeof(float), cudaMemcpyHostToDevice));
    cudaCallMaxXYKernel(blocks, threadsPerBlock, dev_coords, pts.size * 2,
                        dev_max_x, dev_max_y);
    CUDA_KERNEL_CHECK();
    cudaCallMinXYKernel(blocks, threadsPerBlock, dev_coords, pts.size * 2,
                        dev_min_x, dev_min_y);
    CUDA_KERNEL_CHECK();
    CUDA_CALL(cudaMemcpy(&bbox.max_x, dev_max_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&bbox.max_y, dev_max_y, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&bbox.min_x, dev_min_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&bbox.min_y, dev_min_y, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(dev_max_x));
    CUDA_CALL(cudaFree(dev_max_y));
    CUDA_CALL(cudaFree(dev_min_x));
    CUDA_CALL(cudaFree(dev_min_y));
    return bbox;
}
