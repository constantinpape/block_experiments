import json
import pickle
import time
from concurrent import futures
from functools import partial

# the good old anti-segfault vigra import ...
import vigra
import numpy as np
import nifty
import nifty.graph.rag as nrag
import z5py
import cremi_tools.segmentation as cseg


def segment_block(path, out_key, blocking, block_id, halo, segmenter):
    print("Processing block", block_id, "/", blocking.numberOfBlocks)
    t0 = time.time()
    f = z5py.File(path)
    block = blocking.getBlockWithHalo(block_id, halo)

    # all bounding boxes
    bb = tuple(slice(start, stop)
               for start, stop in zip(block.outerBlock.begin, block.outerBlock.end))
    inner_bb = tuple(slice(start, stop)
                     for start, stop in zip(block.innerBlock.begin, block.innerBlock.end))
    local_bb = tuple(slice(start, stop)
                     for start, stop in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

    # load mask
    mask = f['masks/min_filter_mask'][bb]
    # if we only have mask, continue
    if np.sum(mask) == 0:
        print("Finished", block_id, "; contained only mask")
        return time.time() - t0

    # load affinities and convert them to proper format
    affs = f['predictions/full_affs'][(slice(None),) + bb]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    # make watershed
    wslr = cseg.LRAffinityWatershed(threshold_cc=0.1, threshold_dt=0.2, sigma_seeds=2.,
                                    size_filter=50)
    ws, max_label = wslr(affs, mask)

    segmentation = segmenter(ws, affs, max_label + 1)
    ds_out = f[out_key]

    ds_out[inner_bb] = segmentation[local_bb].astype('uint64')
    print("Finished", block_id)
    return time.time() - t0


def segment_mc(ws, affs, n_labels, offsets):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    probs = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets, numberOfThreads=1)[:, 0]
    uv_ids = rag.uvIds()

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = 5 * costs.min()

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def segment_mcrf(ws, affs, offsets, rf):
    pass


def segment_lmc(ws, affs, offsets):
    pass


# TODO also try with single rf learned from both features
def segment_lmcrf(ws, affs, offsets, rf_local, rf_lifted):
    pass


# TODO refactor into function
if __name__ == '__main__':
    path = '/nrs/saalfeld/lauritzen/02/workspace.n5/filtered'
    f = z5py.File(path)

    chunk_shape = (25, 256, 256)
    # TODO maybe use 1k blocks (factor 4) ?!
    block_shape = tuple(cs * 2 for cs in chunk_shape)
    halo = [5, 50, 50]

    shape = f['gray'].shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    segmenter = partial(segment_mc, offsets=offsets)
    n_threads = 60

    group = 'segmentations'
    # CHANGE FOR DIFFERENT SEGMENTERS
    key = 'mc_affs_not_stitched'
    if group not in f:
        f.create_group(group)

    out_key = '%s/%s' % (group, key)
    if out_key not in f:
        f.create_dataset(out_key, shape=shape, chunks=chunk_shape,
                         compression='gzip', dtype='uint64')

    t0 = time.time()
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(segment_block, path, out_key, blocking, block_id, halo, segmenter)
                 for block_id in range(blocking.numberOfBlocks)]
        results = [t.result() for t in tasks]
    # results = [segment_block(path, out_key, blocking, block_id, halo, segmenter)
    #            for block_id in range(blocking.numberOfBlocks)]
    print("Total processing time:", time.time() - t0)

    with open('./timings_02_%s.json' % key, 'w') as ft:
        json.dump(results, ft)
