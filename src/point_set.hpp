#pragma once

typedef struct BBox {
    float min_x;
    float min_y;
    float max_x;
    float max_y;
} BBox;

class PointSet {
public:
    int size;
    float *data;
    PointSet(int size);
    ~PointSet();
    
    inline float get_x(int id) { return data[id*2]; }
    inline float get_y(int id) { return data[id*2 + 1]; }
    inline void set(int id, float xval, float yval) {
        data[id*2] = xval;
        data[id*2 + 1] = yval;
    }
    void resize(int new_size);
    inline float dist_sq(int id1, int id2) {
        return (data[id1*2] - data[id2*2]) * (data[id1*2] - data[id2*2]) +
            (data[id1*2+1] - data[id2*2+1]) * (data[id1*2+1] - data[id2*2+1]);
    }
    BBox extent();
    
};
