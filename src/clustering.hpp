#pragma once
#include <assert.h>
#include <cstdlib>
#include <cstring>

class Clustering {
    /**
     * Store the labels of a clustering by index (which should correspond 
     * to the index in a PointSet).
     *
     * Internally, noise points have label -1, core points have a 
     * positive integer id (unique to its cluster) and border points
     * have an integer id whose value represents
     * the cluster the point is assigned to (though it may not be unique).
     * The cluster label 0 is reserved for unlabeled points.
     */
public:
    int size;
    int *labels;
    bool *border;
    Clustering(int size) {
        this->size = size;
        labels = (int *) malloc(size * sizeof(int));
        border = (bool *) malloc(size * sizeof(bool));
        assert(labels != NULL);
        assert(border != NULL);
        memset(labels, 0, size * sizeof(int));
        memset(border, 0, size * sizeof(bool));
    }
    ~Clustering() {
        free(labels);
        free(border);
    }
    // Getters
    inline int get_cluster(int id) { return labels[id]; }
    inline bool is_noise(int id) { return labels[id] == -1; }
    inline bool is_border(int id) { return border[id]; }
    inline bool is_labeled(int id) { return labels[id] != 0; }
    // Setters
    inline void set_cluster(int id, int cluster) { labels[id] = cluster; }
    inline void set_noise(int id) { labels[id] = -1; }
    inline void set_border(int id, int cluster) {
        labels[id] = cluster;
        border[id] = true;
    }
};
