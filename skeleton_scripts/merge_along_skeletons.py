import os
import numpy as np
import vigra

import nifty
import nifty.graph.rag as nrag
import nifty.z5 as nz5

import z5py
from cremi_tools.skeletons import build_skeleton_metrics


# TODO implement more fancy scheme:
# check whether merging along false splits worsens the false merge score
# and only accept the merge if it does
def merge_along_skeletons(block_id, in_key, out_key, n_threads):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    label_file = os.path.join(path, 'filtered', 'segmentations', in_key)

    # find false splits according to skeletons and the nodes that have to
    # be merged to fix it
    skeleton_file = os.path.join(path, 'skeletons')
    metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)
    skeleton_merges = metrics.mergeFalseSplitNodes(n_threads)

    # TODO can we get max label from attrs like this ?
    n_labels = z5py.File(label_file).attrs['maxId'] + 1

    # get new node labeling with ufd
    ufd = nifty.ufd.ufd(n_labels)
    for _, merge_nodes in skeleton_merges.items():
        merge_nodes = np.array([mn for mn in merge_nodes])
        ufd.merge(merge_nodes)
    node_labels = ufd.elementLabeling()
    # TODO make sure 0 is mapped to zero
    vigra.analysis.relabelConsecutive(node_labels, out=node_labels, keep_zeros=True, start_label=1)

    rag = nrag.gridRagZ5(label_file, numberOfLabels=int(n_labels),
                         numberOfThreads=n_threads, dtype='uint64',
                         blockShape=[25, 256, 256])
    # TODO does this properly create a non-existing dataset ? what about compression ?
    out = nz5.DatasetWrapper('uint64', label_file)
    nrag.projectScalarNodeDataToPixels(rag, node_labels, out, numberOfThreads=n_threads)


if __name__ == '__main__':
    block_id = 2
    in_key = 'multicut_more_features'
    # seg_key = 'wsdt_mc_affs_stitched'
    n_threads = 40
    out_key = 'mc_more_features_merged'

    merge_along_skeletons(block_id, in_key, out_key)
