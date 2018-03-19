import time
import os
from cremi_tools.skeletons import build_skeleton_metrics


def false_split_score(metrics):
    pass


def evaluate_segmentation(block_id, seg_key, n_threads):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    label_file = os.path.join(path, 'filtered', 'segmentations', seg_key)
    skeleton_file = os.path.join(path, 'skeletons')

    t0 = time.time()
    print("Building skeleton metrics in...")
    metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)
    print("... in %f s" % (time.time() - t0,))

    false_split_score(metrics)


if __name__ == '__main__':
    block_id = 2
    seg_key = 'mc_affs_stitched'
    n_threads = 40
    evaluate_segmentation(block_id, seg_key, n_threads)
