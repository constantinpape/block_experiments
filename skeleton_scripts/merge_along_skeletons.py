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
    key1 = '/'.join(('filtered', 'segmentations', in_key))
    label_file = os.path.join(path, key1)

    # find false splits according to skeletons and the nodes that have to
    # be merged to fix it
    skeleton_file = os.path.join(path, 'skeletons')
    metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)
    skeleton_merges = metrics.mergeFalseSplitNodes(n_threads)

    n_labels = z5py.File(label_file).attrs['maxId'] + 1

    # get new node labeling with ufd
    ufd = nifty.ufd.ufd(n_labels)
    for _, merge_nodes in skeleton_merges.items():
        merge_nodes = np.array([mn for mn in merge_nodes])
        ufd.merge(merge_nodes)
    node_labels = ufd.elementLabeling()
    # TODO make sure 0 is mapped to zero
    vigra.analysis.relabelConsecutive(node_labels, out=node_labels, keep_zeros=True, start_label=1)

    labels = nz5.datasetWrapper('uint64', label_file)
    block_shape = [25, 256, 256]
    rag_file = './rag.npy'
    if not os.path.exists(rag_file):
        print("Computing RAG...")
        rag = nrag.gridRagZ5(labels, numberOfLabels=int(n_labels),
                             numberOfThreads=n_threads, dtype='uint64',
                             blockShape=block_shape)
        np.save(rag_file, rag.serialize())
        print("... done")
    else:
        ragser = np.load(rag_file)
        rag = nrag.gridRagZ5(labels, numberOfLabels=int(n_labels),
                             serialization=ragser, dtype='uint64')

    f_out = z5py.File(path)
    key2 = '/'.join(('filtered', 'segmentations', out_key))
    if key2 not in f_out:
        f_out.create_dataset(key2, dtype='uint64', compression='gzip',
                             shape=z5py.File(path)[key1].shape,
                             chunks=z5py.File(path)[key1].chunks)

    out_file = os.path.join(path, key2)
    out = nz5.datasetWrapper('uint64', out_file)

    print("Projecting to pixels...")
    nrag.projectScalarNodeDataToPixels(graph=rag,
                                       nodeData=node_labels,
                                       pixelData=out,
                                       blockShape=block_shape,
                                       numberOfThreads=n_threads)
    print("... done")
    z5py.File(path)[key2].attrs['maxId'] = n_labels - 1


if __name__ == '__main__':
    block_id = 2
    in_key = 'multicut_more_features'
    # seg_key = 'wsdt_mc_affs_stitched'
    n_threads = 40
    out_key = 'mc_more_features_merged'

    merge_along_skeletons(block_id, in_key, out_key, 40)
