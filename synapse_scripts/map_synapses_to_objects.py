import os
import time
import pickle
from concurrent import futures

import numpy as np
import nifty
import z5py


# TODO run connected components on synapses


def get_synapses_to_objects(seg_path, seg_key, syn_path, syn_key,
                            overlap_threshold=50, n_threads=20):
    ds_seg = z5py.File(seg_path)[seg_key]
    ds_syn = z5py.File(syn_path)[syn_key]
    shape = ds_seg.shape
    assert ds_syn.shape == shape
    # TODO parameter
    block_shape = (50, 512, 512)

    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    def map_in_chunk(block_id):
        print("Map synapse to objects for block", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        seg = ds_seg[bb]
        syn = ds_syn[bb]
        syn_mask = syn != 0

        # TODO this could be done more efficiently
        seg = seg[syn_mask]
        syn = syn[syn_mask]

        syn_ids = np.unique(syn)
        synapses_to_objects = {}
        for syn_id in syn_ids:

            object_ids, overlaps = np.unique(seg[syn == syn_id], return_counts=True)
            if object_ids[0] == 0:
                object_ids = object_ids[1:]
                overlaps = overlaps[1:]
            synapses_to_objects[syn_id] = object_ids[overlaps > overlap_threshold]

        return synapses_to_objects

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(map_in_chunk, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
        results = [t.result() for t in tasks]

    synapses_to_objects = {}
    for res in results:
        for syn_id, object_ids in res.items():
            if syn_id in synapses_to_objects:
                synapses_to_objects[syn_id] = np.unique(np.concatenate([object_ids,
                                                                        synapses_to_objects[syn_id]],
                                                                       axis=0))
            else:
                synapses_to_objects[syn_id] = object_ids
    return synapses_to_objects


def get_objects_to_synapses(synapses_to_objects):
    objects_to_synapses = {}
    for syn_id, object_ids in synapses_to_objects.items():
        for object_id in object_ids:
            if object_id in objects_to_synapses:
                objects_to_synapses[object_id].append(syn_id)
            else:
                objects_to_synapses[object_id] = [syn_id]
    return objects_to_synapses


def count_synapses_per_object(objects_to_synapses):
    return {obj: len(syns) for obj, syns in objects_to_synapses.items()}


def extract_undirected_connectivity(objects_to_synapses, synapses_to_objects):
    n_objs = int(max(objects_to_synapses.keys())) + 1
    undirected_connectivity = nifty.graph.undirectedGraph(n_objs)

    edge_weights = {}

    for obj_id, syns in objects_to_synapses.items():

        for syn_id in syns:
            neighbors = synapses_to_objects[syn_id]
            for ngb in neighbors:
                if ngb == obj_id:
                    continue

                edge_id = undirected_connectivity.insertEdge(min(obj_id, ngb), max(obj_id, ngb))
                if obj_id < ngb:
                    if edge_id in edge_weights:
                        edge_weights[edge_id] += 1
                    else:
                        edge_weights[edge_id] = 1

    assert len(edge_weights) == undirected_connectivity.numberOfEdges
    edge_weights = np.array([edge_weights[ii]
                             for ii in range(undirected_connectivity.numberOfEdges)])
    return undirected_connectivity, edge_weights


if __name__ == '__main__':
    seg_path = '/nrs/saalfeld/lauritzen/02/workspace.n5/filtered'
    seg_key = 'segmentations/multicut_more_features'
    syn_path = '/nrs/saalfeld/lauritzen/02/workspace.n5'
    syn_key = 'syncleft_dist_DTU-2_200000_cc'

    # Extract synapses to objects
    cache_path = 'synapses_to_objects.pkl'
    if not os.path.exists(cache_path):
        t0 = time.time()
        syns_to_objs = get_synapses_to_objects(seg_path, seg_key, syn_path, syn_key)
        print("Extracting synapses to objects in %f s" % (time.time() - t0))
        with open(cache_path, 'wb') as f:
            pickle.dump(syns_to_objs, f)

    else:
        with open(cache_path, 'rb') as f:
            syns_to_objs = pickle.load(f)

    # Extract objects to synapses
    objs_to_syns = get_objects_to_synapses(syns_to_objs)

    # Extract synapse counts
    print("Mean synapse count (of objects with synapse)")
    syn_counts = count_synapses_per_object(objs_to_syns)
    print(np.mean(list(syn_counts.values())))
    print()

    # Extract (undirected) connectivity
    connectivity, edge_weights = extract_undirected_connectivity(objs_to_syns, syns_to_objs)
    print("Undirected connectivity has", connectivity.numberOfEdges, "edges")
    print("Mean edge strength:", np.mean(edge_weights))
