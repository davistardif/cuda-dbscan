#include <vector>

#include "point_set.hpp"
#include "clustering.hpp"
#include "cpu_dbscan.hpp"
using namespace std;

/**
 * Return a list of ids who are within distance epsilon 
 * (calculated by Euclidean distance) of the specified 
 * point id. Does not include id in the list.
 */
vector<int> neighbors(int id, PointSet &pts, float epsilon) {
    vector<int> v;
    float eps_sq = epsilon * epsilon;
    for (int i = 0; i < pts.size; i++) {
        if (i == id)
            continue;
        if (pts.dist_sq(id, i) <= eps_sq) {
            v.push_back(i);
        }
    }
    return v;       
}

Clustering naive_dbscan(PointSet &pts, float epsilon, unsigned int min_points) {
    Clustering clusters(pts.size);
    int cluster_count = 0;
    for (int i = 0; i < pts.size; i++) {
        if (clusters.is_labeled(i))
            continue;
        vector<int> nbrs = neighbors(i, pts, epsilon);
        if (nbrs.size() + 1 < min_points) {
            // note neighbor set also excludes the point we're searching from
            // hence the + 1
            clusters.set_noise(i);
            continue;
        }
        cluster_count += 1;
        clusters.set_cluster(i, cluster_count);
        for (int j = 0; j < nbrs.size(); j++) {
            int id = nbrs[j];
            if (clusters.is_noise(id)) {
                clusters.set_border(id, cluster_count);
                continue;
            }
            else if (clusters.is_labeled(id))
                continue;
            
            vector<int> new_nbrs = neighbors(id, pts, epsilon);
            if (new_nbrs.size() + 1 >= min_points) {
                clusters.set_cluster(id, cluster_count);
                nbrs.reserve(nbrs.size() + new_nbrs.size());
                nbrs.insert(nbrs.end(), new_nbrs.begin(), new_nbrs.end());
            }
            else {
                clusters.set_border(id, cluster_count);
            }
        }
        
    }
    return clusters;
}
