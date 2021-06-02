#pragma once

void callGridLabelKernel(uint blocks, uint threadsPerBlock,
                         uint *dev_pt_ids, uint *dev_grid_labels,
                         float *dev_coords,
                         float min_x, float min_y, float side_len,
                         uint grid_x_size, int num);

void callGridMarkCoreCells(uint blocks, uint threadsPerBlock,
                           uint *d_index_counts, uint unique_key_count,
                           uint *d_values, bool *isCore, uint min_points);

void callGridCheckCore(float *dev_coords, uint *d_index_counts,
                       uint key_count, uint *d_values, bool *d_isCore,
                       uint min_points, float EPS_SQ, float x, float y,
                       int pt_idx);
