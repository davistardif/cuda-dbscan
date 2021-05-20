#include "disjoint_set.hpp"

#include <cassert>
#include <cstdlib>

DisjointSet::DisjointSet(int max_elems) {
    parent = (int *) malloc(max_elems * sizeof(int));
    rank = (int *) malloc(max_elems * sizeof(int));
    assert(parent != nullptr);
    assert(rank != nullptr);
    for (int i = 0; i < max_elems; i++) {
        make_set(i);
    }
}

DisjointSet::~DisjointSet() {
    free(parent);
}

void DisjointSet::make_set(int v) {
    parent[v] = v;
    rank[v] = 0;
}

int DisjointSet::find_set(int v) {
    if (v == parent[v])
        return v;
    return parent[v] = find_set(parent[v]);
}

void DisjointSet::union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (rank[a] < rank[b]) {
            int temp = a;
            a = b;
            b = temp;
        }
        parent[b] = a;
        if (rank[a] == rank[b])
            rank[a]++;
    }
}
