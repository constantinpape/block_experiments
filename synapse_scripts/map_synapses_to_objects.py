from concurrent import futures
import time
import pickle

import numpy as np
import nifty
import z5py


# TODO run connected components on synapses


def get_synapses_to_objects(seg_path, seg_key, syn_path, syn_key,
                            overlap_threshold, n_threads):
    ds_seg = z5py.File(seg_path)[seg_key]
    ds_syn = z5py.File(syn_path)[syn_key]
    shape = ds_seg.shape
    assert ds_syn.shape == shape
    chunks = ds_seg.chunks
    # TODO it would be sufficient if these are factors of each other
    assert ds_syn.chunks == chunks

    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(chunks))

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


def count_synapses_per_object():
    pass


def extract_undirected_connectivity():
    pass


if __name__ == '__main__':
    seg_path = '/nrs/saalfeld/lauritzen/02/workspace.n5/filtered'
    seg_key = 'segmentations/multicut_more_features'
    syn_path = '/nrs/saalfeld/lauritzen/02/workspace.n5'
    syn_key = 'syncleft_dist_DTU-2_200000_cc'

    t0 = time.time()
    syns_to_objs = get_synapses_to_objects(seg_path, seg_key, syn_path, syn_key, 8)
    print("Extracting synapses to objects in %f s" % (time.time() - t0))
    with open('synapses_to_objects.pkl', 'wb') as f:
        pickle.dump(syns_to_objs, f)
