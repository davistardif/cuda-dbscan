#pragma once
#include <assert.h>
#include <cstdlib>
#include <cstring>

class PointSet {
public:
    int size;
    float *data;
    PointSet(int size) {
        this->size = size;
        data = (float *) malloc(2 * size * sizeof(float));
        assert(data != NULL);
        memset(data, 0, 2 * size * sizeof(float));
    }
    ~PointSet() { free(data); }
    
    inline float get_x(int id) { return data[id*2]; }
    inline float get_y(int id) { return data[id*2 + 1]; }
    void set(int id, float xval, float yval) {
        data[id*2] = xval;
        data[id*2 + 1] = yval;
    }
    void resize(int new_size) {
        data = (float *) realloc(data, new_size * 2 * sizeof(float));
        size = new_size;
    }
    
};
