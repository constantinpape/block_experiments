import os
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


# TODO return the max id of the multicut to find proper offsets
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
        return 0, time.time() - t0

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

    segmentation = segmentation[local_bb].astype('uint64')
    segmentation, max_id, _ = vigra.analysis.relabelConsecutive(segmentation, start_label=1)
    segmentation[mask] = 0

    ds_out[inner_bb] = segmentation
    print("Finished", block_id)
    return max_id, time.time() - t0


def segment_mc(ws, affs, n_labels, offsets, return_nodes=False):
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
    if return_nodes:
        return node_labels
    else:
        return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def segment_mcrf(ws, affs, n_labels, offsets, rf, return_nodes=False):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    feats = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets, numberOfThreads=1)
    probs = rf.predict_proba(feats)[:, 1]
    uv_ids = rag.uvIds()

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = 5 * costs.min()

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)
    if return_nodes:
        return node_labels
    else:
        return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def segment_lmc(ws, affs, n_labels, offsets, return_nodes=False):
    pass


# TODO also try with single rf learned from both features
def segment_lmcrf(ws, affs, n_labels, offsets, rf_local, rf_lifted, return_nodes=False):
    pass


def run_segmentation(block_id, segmenter, key, n_threads=60):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/filtered' % block_id
    f = z5py.File(path)

    chunk_shape = (25, 256, 256)
    # TODO maybe use 1k blocks (factor 4) ?!
    block_shape = tuple(cs * 2 for cs in chunk_shape)
    halo = [5, 50, 50]

    shape = f['gray'].shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    group = 'segmentations'
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
    # block_times = [segment_block(path, out_key, blocking, block_id, halo, segmenter)
    #                for block_id in range(blocking.numberOfBlocks)]
    t_tot = time.time() - t0
    print("Total processing time:", t_tot)

    times = {'blocks': [res[1] for res in results], 'total': t_tot}
    with open('./timings_0%i_%s.json' % (block_id, key), 'w') as ft:
        json.dump(times, ft)

    offsets = np.array([res[0] for res in results], dtype='uint64')
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    with open('./block_offsets_0%i_%s.json' % (block_id, key), 'w') as ft:
        json.dump(offsets, ft)


def segmenter_factory(algo, feats, return_nodes=False):
    rf_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/random_forests'
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]

    if algo == 'mc':

        if feats == 'local':
            # affinity based mc
            key = 'mc_affs_not_stitched'
            segmenter = partial(segment_mc, offsets=offsets, return_nodes=return_nodes)

        elif feats == 'rf':
            # rf based mc
            with open(os.path.join(rf_folder, 'rf_ABC_local_affinity_feats.pkl'), 'rb') as fr:
                rf = pickle.load(fr)
            rf.n_jobs = 1
            key = 'mc_rf_not_stitched'
            segmenter = partial(segment_mcrf, offsets=offsets, rf=rf, return_nodes=return_nodes)

        else:
            raise AttributeError("No!")

    elif algo == 'lmc':

        if feats == 'local':
            # affinity based lmc
            key = 'lmc_affs_not_stitched'
            segmenter = partial(segment_lmc, offsets=offsets, return_nodes=return_nodes)

        elif feats == 'rf':
            # rf based lmc
            with open(os.path.join(rf_folder, 'rf_ABC_local_affinity_feats.pkl'), 'rb') as fr:
                rf1 = pickle.load(fr)
            rf1.n_jobs = 1
            with open(os.path.join(rf_folder, 'rf_ABC_lifted_affinity_feats.pkl'), 'rb') as fr:
                rf2 = pickle.load(fr)
            rf2.n_jobs = 1

            key = 'lmc_rf_not_stitched'
            segmenter = partial(segment_lmcrf, offsets=offsets,
                                rf_local=rf1, rf_lifted=rf2, return_nodes=return_nodes)

        else:
            raise AttributeError("No!")

    else:
        raise AttributeError("No!")
    return key, segmenter


if __name__ == '__main__':
    block_id = 2
    key, segmenter = segmenter_factory('mc', 'rf')
    n_threads = 60
    run_segmentation(block_id, segmenter, key, n_threads)
