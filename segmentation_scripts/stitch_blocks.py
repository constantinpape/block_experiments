# import os
import json
from concurrent import futures
from itertools import chain
import numpy as np

import vigra
import nifty
import z5py
from blockwise_segmentations import segmenter_factory


def stitch_block_neighbors(ds_seg, ds_affs, blocking, block_id, halo, segmenter, offsets):

    # don't stitch with empty blocks
    if offsets[block_id] == 0 and block_id > 0:
        return []
    print("Stitching block", block_id, "/", blocking.numberOfBlocks)

    block = blocking.getBlockWithHalo(block_id, halo)
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
        if offsets[ngb_id] == 0:
            continue

        # find the overlap
        overlap_bb = tuple(slice(inner_block.begin[i], inner_block.end[i]) if i != dim else
                           slice(inner_block.end[i] - halo[i], outer_block.end[i])
                           for i in range(3))
        bb_offset = tuple(ovlp.start for ovlp in overlap_bb)

        # load segmentation and affinities for the overlap
        seg = ds_seg[overlap_bb]
        affs = ds_affs[(slice(None),) + overlap_bb]

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 AAA")
        # print(seg.shape)
        # find the parts off the overlap associated with this block and the neighbor block
        ngb_block = blocking.getBlockWithHalo(ngb_id, halo)
        bb_a = tuple(slice(inner_block.begin[i] - off, inner_block.end[i] - off) if i != dim else
                     slice(inner_block.end[i] - off, outer_block.end[i] - off)
                     for i, off in enumerate(bb_offset))
        inner_ngb, outer_ngb = ngb_block.innerBlock, ngb_block.outerBlock
        bb_b = tuple(slice(inner_ngb.begin[i] - off, inner_ngb.end[i] - off) if i != dim else
                     slice(outer_ngb.begin[i] - off, inner_ngb.begin[i] - off)
                     for i, off in enumerate(bb_offset))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 BBB")

        # add proper offsets to the 2 segmentation halves
        # (but keep mask !)
        bg_mask = seg == 0
        seg[bb_a] += offsets[block_id]
        seg[bb_b] += offsets[ngb_id]
        seg[bg_mask] = 0
        print("Stitching with", ngb_id)
        print(np.unique(seg))

        # TODO restrict merges to segments that actually touch at the overlap surface ??
        # run segmenter to get the node assignment
        n_labels = int(seg.max()) + 1
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 CCC")
        merged_nodes = segmenter(seg, affs, n_labels)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 DDD")
        if merged_nodes.size > 0:
            node_assignments.append(merged_nodes)

    print("Finished block", block_id)
    return np.array(node_assignments) if node_assignments else []


# TODO we might want to overwrite the non-stitched segmentation later.
def write_stitched_segmentation(block_id, blocking, ds, ds_out, node_labels, offsets):
    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    seg = ds[bb]
    seg += offsets[block_id]
    seg = nifty.tools.take(node_labels, seg)
    ds_out[bb] = seg


def stitch_segmentation(block_id, key, segmenter, n_threads=40):
    offsets = './block_offsets_0%i_%s.json' % (block_id, key)
    with open(offsets, 'r') as f:
        offsets = json.load(f)
    # add up the offsets
    # ignore blocks which have only background (offset = zero)
    bg_blocks = offsets == 0
    last_offset = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    n_labels = offsets[-1] + last_offset
    offsets[bg_blocks] = 0

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
    in_key = '%s/%s' % (group, key)
    ds = f[in_key]
    ds_affs = f['predictions/full_affs']

    # find the node assignmemnts by running the segmenter on the (segmentation) overlaps
    # TODO also need to pass the affinity dataset as an argument
    # TODO parallelize

    results = [stitch_block_neighbors(ds, ds_affs, blocking, block_id,
                                      halo, segmenter, offsets)
               for block_id in range(blocking.numberOfBlocks)]
    results = list(chain.from_iterable(results))

    # with futures.ThreadPoolExecutor(n_threads) as tp:
    #     tasks = [tp.submit(stitch_block_neighbors, ds, ds_affs, blocking, block_id,
    #                        halo, segmenter, offsets)
    #              for block_id in range(blocking.numberOfBlocks)]
    #     results = list(chain.from_iterable([t.result() for t in tasks]))

    results = np.array([res for res in results if res])

    # cache for debugging
    print(results.shape)
    np.save('./node_assignment.npy', results)

    # stitch the segmentation (node level)
    ufd = nifty.ufd.ufd(n_labels)
    ufd.merge(results)
    node_labels = ufd.elementLabeling()
    vigra.analysis.relabelConsecutive(node_labels, keep_zeros=True, start_label=1)

    # save the stitched result
    out_key = key.split('_')
    out_key.remove('not')
    out_key = '_'.join(out_key)
    out_key = '%s/%s' % (group, out_key)
    ds_out = f.create_dataset(out_key, shape=ds.shape,
                              chunks=chunk_shape, dtype='uint64', compression='gzip')

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(write_stitched_segmentation, block_id, blocking, ds,
                           ds_out, node_labels, offsets)
                 for block_id in range(blocking.numberOfBlocks)]
        [t.result() for t in tasks]


if __name__ == '__main__':
    block_id = 2
    key, segmenter = segmenter_factory('mc', 'local', return_merged_nodes=True)
    stitch_segmentation(block_id, key, segmenter)
