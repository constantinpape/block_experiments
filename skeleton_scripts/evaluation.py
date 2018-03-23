import time
import os
import pickle

import numpy as np
from cremi_tools.skeletons import build_skeleton_metrics


def false_split_score(metrics):

    # node_assignments = metrics.getNodeAssignments()
    # for skel_id, assignments in node_assignments.items():
    #     print(skel_id, len(assignments))
    #     print(np.unique(list(assignments.values())))

    print("Split scores:")
    split_score = metrics.computeSplitScores()
    for skel_id, val in split_score.items():
        print("Skeleton", skel_id, ":", val)

    # TODO meaningful representation of runlens
    resolution = [40., 4., 4.]
    skel_runlens, frag_runlens = metrics.computeSplitRunlengths(resolution)
    assert len(skel_runlens) == len(split_score), "%i, %i" % (len(skel_runlens), len(split_score))

    # for sk_id, val in skel_runlens.items():
    #     print("Total runlen of skeleton", sk_id, ":", val, "nm")
    #     # print("(split-score:)", split_score[sk_id])
    #     for f_id, fval in frag_runlens[sk_id].items():
    #         print("Segmentation fragment", f_id, "has runlen", fval, "nm (%.2f percent)" % (fval / val * 100))


def explicit_merge_score(metrics):
    merges = metrics.computeExplicitMerges()
    if merges:
        print("Explicit merges")
        for skel_id, labels in merges.items():
            print("Skeleton", skel_id, "contains merge with label")
            print(labels)
    else:
        print("No false merges")


# TODO come up with meaningful histograms for this
def distance_statistics(metrics):
    cache_path = './skeleton_distance_cache.pkl'
    if os.path.exists(cache_path):
        print("Loading skeleton distances from cache")
        with open(cache_path, 'rb') as f:
            stats = pickle.load(f)
    else:
        t0 = time.time()
        resolution = [40., 4., 4.]
        print("Computing skeleton distances in ...")
        stats = metrics.computeDistanceStatistics(resolution, numberOfThreads=40)
        print("... %f s" % (time.time() - t0))
        with open(cache_path, 'wb') as f:
            pickle.dump(stats, f)

    for skel_id, values in stats.items():
        print(skel_id, len(values))
        print(np.mean(values[next(iter(values))]))


def evaluate_segmentation(block_id, seg_key, n_threads):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    label_file = os.path.join(path, 'filtered', 'segmentations', seg_key)
    skeleton_file = os.path.join(path, 'skeletons')

    t0 = time.time()
    print("Building skeleton metrics in...")
    metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)
    print("... in %f s" % (time.time() - t0,))

    # false_split_score(metrics)
    # explicit_merge_score(metrics)
    distance_statistics(metrics)


#
def plot_distance_statistics(cache_path='./skeleton_distance_cache.pkl'):
    import matplotlib.pyplot as plt
    with open(cache_path, 'rb') as f:
        stats = pickle.load(f)

    n_bins = 32
    for skel_id, vals in stats.items():
        mean_dists = []
        std_dists = []
        for val in vals:
            mean_dists.append(np.mean(val))
            std_dists.append(np.std(val))
        # TODO plot histogram
        fig, ax = plt.subplots(2)
        ax[0].hist(mean_dists, bins=n_bins)
        ax[1].hist(std_dists, bins=n_bins)
        plt.show()


if __name__ == '__main__':
    block_id = 2
    seg_key = 'multicut_more_features'
    # seg_key = 'wsdt_mc_affs_stitched'
    n_threads = 40
    # evaluate_segmentation(block_id, seg_key, n_threads)
    plot_distance_statistics()
