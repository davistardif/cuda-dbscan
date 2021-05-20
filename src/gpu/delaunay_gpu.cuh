#pragma once

void cudaCallMaxXYKernel(const unsigned int blocks,
                         const unsigned int threadsPerBlock,
                         float *coords, int length, float *max_x, float *max_y);
