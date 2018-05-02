#! /usr/bin/python

import os
import argparse
import json
import pickle
import time
from functools import partial

import vigra
import numpy as np
import nifty
import nifty.graph.rag as nrag
import z5py
import cremi_tools.segmentation as cseg


def segment_block(path, out_key, blocking, block_id,
                  halo, fragmenter, segmenter):
    t0 = time.time()
    # print("Processing block", block_id)
    f = z5py.File(path)
    block = blocking.getBlockWithHalo(block_id, list(halo))

    # all bounding boxes
    bb = tuple(slice(start, stop)
               for start, stop in zip(block.outerBlock.begin, block.outerBlock.end))
    inner_bb = tuple(slice(start, stop)
                     for start, stop in zip(block.innerBlock.begin, block.innerBlock.end))
    local_bb = tuple(slice(start, stop)
                     for start, stop in zip(block.innerBlockLocal.begin,
                                            block.innerBlockLocal.end))

    # load mask
    mask = f['masks/minfilter_mask'][bb].astype('bool')
    # if we only have mask, continue
    if np.sum(mask) == 0:
        # print("All masked", block_id)
        return 0, time.time() - t0

    # load affinities and convert them to proper format
    affs = f['predictions/affs_glia'][(slice(None),) + bb]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.

    # if we have 13 channels, the last one predicts glia
    # and is not inverted:
    n_channels = affs.shape[0]
    if n_channels == 13:
        affs[:-1] = 1. - affs[:-1]
    else:
        affs = 1. - affs

    ws, max_id = fragmenter(affs, mask)
    # TODO we might want to keep glia as extra channel for rf ?!
    # if we have glia predictions, remove them
    if n_channels == 13:
        affs = affs[:-1]
    segmentation = segmenter(ws, affs, max_id + 1, block_id=block_id)
    # for debug purposes
    segmentation = ws
    ds_out = f[out_key]

    segmentation = segmentation[local_bb].astype('uint64')
    segmentation, max_id, _ = vigra.analysis.relabelConsecutive(segmentation, start_label=1)
    mask = mask[local_bb]
    ignore_mask = np.logical_not(mask)
    segmentation[ignore_mask] = 0

    ds_out[inner_bb] = segmentation
    # print("Done block", block_id)
    return max_id, time.time() - t0


def compute_wslr_fragments(affs, mask, channel_weights=None, thresh=0.05):
    wsfu = cseg.LRAffinityWatershed(channel_weights=channel_weights,
                                    threshold_cc=thresh, threshold_dt=0.3,
                                    sigma_seeds=1.6, size_filter=15)
    return wsfu(affs, mask)


# TODO features only based on neareast affinities ?
def compute_mc_segments(ws, affs, n_labels, offsets, block_id):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    uv_ids = rag.uvIds()

    # we can have a single over-segment id for (small border) blocks
    # resulting in 0 edges
    if(uv_ids.shape[0] == 0):
        print("WARNING:, block", block_id, "contains only a single id, but is not masked")
        print("This is may be caused by an incorrect mask")
        return ws

    probs = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets, numberOfThreads=1)[:, 0]

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    # print(probs.shape, uv_ids.shape, costs.shape)
    costs[ignore_edges] = 5 * costs.min()

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


# TODO features only based on neareast affinities ?
def compute_mcrf_segments(ws, affs, n_labels, offsets, rf, block_id):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    uv_ids = rag.uvIds()

    # we can have a single over-segment id for (small border) blocks
    # resulting in 0 edges
    if(uv_ids.shape[0] == 0):
        print("WARNING:, block", block_id, "contains only a single id, but is not masked")
        print("This is may be caused by an incorrect mask")
        return ws

    feats = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets, numberOfThreads=1)
    probs = rf.predict_proba(feats)[:, 1]

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = 5 * costs.min()

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def process_block_list(path, out_key, cache_folder, job_id, block_shape, halo, rf_path):

    input_file = os.path.join(cache_folder, 'block_list_job%i.json' % job_id)
    with open(input_file) as f:
        block_list = json.load(f)

    shape = z5py.File(path)['gray'].shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    fragmenter = compute_wslr_fragments

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    if rf_path == '':
        segmenter = partial(compute_mc_segments, offsets=offsets)
    else:
        assert os.path.exists(rf_path), rf_path
        with open(rf_path, 'rb') as f:
            rf = pickle.load(f)
        segmenter = partial(compute_mcrf_segments, offsets=offsets, rf=rf)

    results = [segment_block(path, out_key, blocking, block_id,
                             halo, fragmenter, segmenter)
               for block_id in block_list]

    max_ids = [res[0] for res in results]
    res_file = os.path.join(cache_folder, 'block_offsets_job%i.json' % job_id)
    with open(res_file, 'w') as f:
        json.dump({block_id: max_id for block_id, max_id in zip(block_list, max_ids)}, f)

    times = [res[1] for res in results]
    print("Success")
    for block_index, tproc in enumerate(times):
        print("Processed block", block_list[block_index], "in", tproc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('--block_shape', type=int, nargs=3)
    parser.add_argument('--halo', type=int, nargs=3)
    parser.add_argument('--rf_path', type=str, default='')

    args = parser.parse_args()
    process_block_list(args.path, args.out_key,
                       args.cache_folder, args.job_id,
                       tuple(args.block_shape),
                       tuple(args.halo), args.rf_path)
