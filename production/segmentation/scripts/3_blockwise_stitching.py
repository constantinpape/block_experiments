#! /usr/bin/python

import argparse
import pickle
import os
import time
import json
import numpy as np
from functools import partial

import vigra
import nifty
import nifty.graph.rag as nrag
import z5py
import cremi_tools.segmentation as cseg


def stitch_block_with_neighbors(ds_seg, ds_affs,
                                blocking, block_id,
                                halo, segmenter,
                                empty_blocks):
                                # offsets, empty_blocks):

    t0 = time.time()
    block = blocking.getBlockWithHalo(block_id, list(halo))
    inner_block, outer_block = block.innerBlock, block.outerBlock

    node_assignments = []

    # iterate over the (upper) neighbors of this block and stitch with the segmenter
    to_lower = False
    for dim in range(3):

        # find the block id of the overlapping neighbor
        ngb_id = blocking.getNeighborId(block_id, dim, to_lower)
        if ngb_id == -1:
            continue

        # don't stitch with empty blocks
        if ngb_id in empty_blocks:
            continue

        # find the overlap
        overlap_bb = tuple(slice(inner_block.begin[i], inner_block.end[i]) if i != dim else
                           slice(inner_block.end[i] - halo[i], outer_block.end[i])
                           for i in range(3))
        # bb_offset = tuple(ovlp.start for ovlp in overlap_bb)

        # load segmentation and affinities for the overlap
        seg = ds_seg[overlap_bb]

        # find the parts off the overlap associated with this block and the neighbor block
        # bb_a = tuple(slice(inner_block.begin[i] - off, inner_block.end[i] - off) if i != dim else
        #              slice(inner_block.end[i] - off, outer_block.end[i] - off)
        #              for i, off in enumerate(bb_offset))

        # ngb_block = blocking.getBlockWithHalo(ngb_id, list(halo))
        # inner_ngb, outer_ngb = ngb_block.innerBlock, ngb_block.outerBlock
        # bb_b = tuple(slice(inner_ngb.begin[i] - off, inner_ngb.end[i] - off) if i != dim else
        #              slice(outer_ngb.begin[i] - off, inner_ngb.begin[i] - off)
        #              for i, off in enumerate(bb_offset))

        # add proper offsets to the 2 segmentation halves
        # (but keep mask !)
        bg_mask = seg == 0

        # continue if we have only background in the overlap
        if np.sum(bg_mask) == bg_mask.size:
            continue

        affs = ds_affs[(slice(None),) + overlap_bb]
        if affs.dtype == np.dtype('uint8'):
            affs = affs.astype('float32') / 255.
        glia = affs[-1]
        affs = affs[:-1]
        affs = 1. - affs

        # TODO we try merging without on the fly offsets, and do it in a seperate
        # step instead
        # # FIXME this is exactly the opposite of what I have expected ?!
        # seg[bb_a] += offsets[ngb_id]
        # seg[bb_b] += offsets[block_id]
        # seg[bg_mask] = 0

        # TODO restrict merges to segments that actually touch at the overlap surface ??
        # run segmenter to get the node assignment
        n_labels = int(seg.max()) + 1
        merged_nodes = segmenter(seg, affs, n_labels, glia=glia)

        if merged_nodes.size > 0:
            node_assignments.append(merged_nodes)

    return np.concatenate(node_assignments, axis=0) if node_assignments else None, time.time() - t0


def get_merged_nodes(uv_ids, node_labels):
    merge_edges = node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]]
    merge_nodes = uv_ids[merge_edges]
    # for some reasons that may still happen, although zero should be masked
    valid_merges = (merge_nodes != 0).all(axis=1)
    return merge_nodes[valid_merges]


def compute_mc_nodes(ws, affs, n_labels, offsets, glia=None):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    probs = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets, numberOfThreads=1)[:, 0]
    uv_ids = rag.uvIds()

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = - 100

    # run multicut
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)
    # find the pairs of merged nodes from the multicut
    # node labeling
    return get_merged_nodes(uv_ids, node_labels)


def glia_features(rag, seg, glia):
    uv_ids = rag.uvIds()
    # FIXME for some reason 'Quantiles' are not working
    statistics = ["Mean", "Variance", "Skewness", "Kurtosis",
                  "Minimum", "Maximum", "Count", "RegionRadii"]
    extractor = vigra.analysis.extractRegionFeatures(glia, seg.astype('uint32'),
                                                     features=statistics)

    node_features = np.concatenate([extractor[stat_name][:, None].astype('float32')
                                    if extractor[stat_name].ndim == 1
                                    else extractor[stat_name].astype('float32')
                                    for stat_name in statistics],
                                   axis=1)
    fU = node_features[uv_ids[:, 0], :]
    fV = node_features[uv_ids[:, 1], :]

    edge_features = [np.minimum(fU, fV),
                     np.maximum(fU, fV),
                     np.abs(fU - fV),
                     fU + fV]
    edge_features = np.concatenate(edge_features, axis=1)
    return np.nan_to_num(edge_features)


def compute_mcrf_nodes(ws, affs, n_labels, offsets, glia, rf):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    # TODO split features over different affinity ranges ?
    feats = np.concatenate([nrag.accumulateAffinityStandartFeatures(rag,
                                                                    affs,
                                                                    offsets,
                                                                    numberOfThreads=1),
                            glia_features(rag, ws, glia)], axis=1)
    probs = rf.predict_proba(feats)[:, 1]
    uv_ids = rag.uvIds()

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = -100

    # run multicut
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)
    # find the pairs of merged nodes from the multicut
    # node labeling
    return get_merged_nodes(uv_ids, node_labels)


def stitch_blocks(path, out_key, cache_folder, job_id, block_shape, halo, rf_path):

    input_file = os.path.join(cache_folder, 'block_list_job%i.json' % job_id)
    with open(input_file) as f:
        block_list = json.load(f)

    shape = z5py.File(path)['gray'].shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    offset_file = os.path.join(cache_folder, 'block_offsets.json')
    with open(offset_file, 'r') as f:
        offsets_dict = json.load(f)
        # block_offsets = offsets_dict['offsets']
        empty_blocks = offsets_dict['empty_blocks']
        max_label = offsets_dict['n_labels'] - 1

    f = z5py.File(path)

    ds_seg = f[out_key]
    ds_affs = f['predictions/affs_glia']

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    if rf_path == '':
        segmenter = partial(compute_mc_nodes, offsets=offsets)
    else:
        assert os.path.exists(rf_path), rf_path
        with open(rf_path, 'rb') as f:
            rf = pickle.load(f)
        segmenter = partial(compute_mcrf_nodes, offsets=offsets, rf=rf)

    # find the node assignmemnts by running the segmenter on the (segmentation) overlaps
    results = [stitch_block_with_neighbors(ds_seg, ds_affs, blocking, block_id,
                                           halo, segmenter, empty_blocks)
                                           # halo, segmenter, block_offsets, empty_blocks)
               for block_id in block_list if block_id not in empty_blocks]

    assignments = [res[0] for res in results if res[0] is not None]
    if assignments:
        assignments = np.concatenate(assignments, axis=0)
    out_path = os.path.join(cache_folder, 'node_assignments_job%i.npy' % job_id)
    np.save(out_path, assignments)

    times = [res[1] for res in results]

    # write out the max-label with job 0
    if job_id == 0:
        ds_seg.attrs['maxId'] = max_label

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
    stitch_blocks(args.path, args.out_key,
                  args.cache_folder, args.job_id,
                  tuple(args.block_shape),
                  tuple(args.halo), args.rf_path)
