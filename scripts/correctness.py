# Usage: python3 correctness.py reference_result.txt result_to_compare.txt
import sys
import heapq

def get_clusters_and_noise(f):
    clusters = {}
    noise = set()
    for row in f:
        pt_id, pt_type, clu_id = row.split()
        pt_id = int(pt_id)
        clu_id = int(clu_id)
        if pt_type == "n":
            noise.add(pt_id)
        elif pt_type == "b":
            # ignore border points for now (TODO)
            continue
        elif clu_id in clusters:
            heapq.heappush(clusters[clu_id], pt_id)
        else:
            clusters[clu_id] = [pt_id]
    return clusters, noise

def clusters_by_min_id(clusters):
    cluster_dict = {}
    for h in clusters.values():
        cluster_dict[h[0]] = h
    return cluster_dict

def check(file1, file2):
    clusters1, noise1 = get_clusters_and_noise(file1)
    clusters2, noise2 = get_clusters_and_noise(file2)
    ndiff = noise1.symmetric_difference(noise2)
    for pt in ndiff:
        if pt in noise1:
            print(f"ERROR: point {pt} should be labeled noise")
        else:
            print(f"ERROR: point {pt} should not be labeled noise")
    clusters_by_min1 = clusters_by_min_id(clusters1)
    clusters_by_min2 = clusters_by_min_id(clusters2)
    for k, v in clusters_by_min1.items():
        if k not in clusters_by_min2:
            print(f"ERROR: point {k} is in the wrong cluster. It should be in {v}")
        else:
            s1 = set(v)
            s2 = set(clusters_by_min2[k])
            diff = s1.symmetric_difference(s2)
            if len(diff) > 0:
                print(f"ERROR: cluster mismatch:\n{list(sorted(v))}\n{list(sorted(clusters_by_min2[k]))}")
    for k, v in clusters_by_min2.items():
        if k not in clusters_by_min1:
            print(f"ERROR: point {k} is in the wrong cluster")

def main():
    if len(sys.argv) != 3:
        print('Usage: python3 correctness.py reference_result.txt result_to_compare.txt')
        return -1
    check(open(sys.argv[1], 'r'), open(sys.argv[2], 'r'))

if __name__ == '__main__':
    main()
