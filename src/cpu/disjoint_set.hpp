#pragma once

class DisjointSet {
public:
    DisjointSet(int max_elems);
    ~DisjointSet();
    void make_set(int v);
    int find_set(int v);
    void union_sets(int a, int b);
private:
    int *parent;
    int *rank;
};
