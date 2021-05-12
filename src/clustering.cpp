#include "clustering.hpp"
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;
Clustering::Clustering(int size) {
    this->size = size;
    labels = (int *) malloc(size * sizeof(int));
    border = (bool *) malloc(size * sizeof(bool));
    assert(labels != NULL);
    assert(border != NULL);
    memset(labels, 0, size * sizeof(int));
    memset(border, 0, size * sizeof(bool));
}

Clustering::~Clustering() {
    free(labels);
    free(border);
}

void Clustering::print() {
    for (int i = 0; i < this->size; i++) {
        char type = (this->is_noise(i) ? 'n' : (
                         this->is_border(i) ? 'b' : 'c'));
        cout << i << " " << type << " " << this->get_cluster(i) << "\n";
    }
}
