#include "point_set.hpp"

#include <assert.h>
#include <cstdlib>
#include <cstring>


PointSet::PointSet(int size) {
    this->size = size;
    data = (float *) malloc(2 * size * sizeof(float));
    assert(data != NULL);
    memset(data, 0, 2 * size * sizeof(float));
}

PointSet::~PointSet() { free(data); }

void PointSet::resize(int new_size) {
    data = (float *) realloc(data, new_size * 2 * sizeof(float));
    size = new_size;
}

BBox PointSet::extent() {
    BBox b;
    b.min_x = get_x(0);
    b.max_x = get_x(0);
    b.min_y = get_y(0);
    b.max_y = get_y(0);
    for (int i = 1; i < size; i++) {
        float x = get_x(i);
        float y = get_y(i);
        if (x < b.min_x) {
            b.min_x = x;
        }
        else if (x > b.max_x) {
            b.max_x = x;
        }
        if (y < b.min_y) {
            b.min_y = y;
        }
        else if (y > b.max_y) {
            b.max_y = y;
        }
    }
    return b;
}
