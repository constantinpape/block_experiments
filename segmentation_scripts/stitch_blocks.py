import time
import json
from concurrent import futures
import numpy as np

import vigra
import nifty
import z5py
from backend import segmenter_factory


def stitch_block_neighbors(ds_seg, ds_affs, blocking, block_id, halo, segmenter, offsets, empty_blocks):

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
        if empty_blocks[ngb_id]:
            continue

        # find the overlap
        overlap_bb = tuple(slice(inner_block.begin[i], inner_block.end[i]) if i != dim else
                           slice(inner_block.end[i] - halo[i], outer_block.end[i])
                           for i in range(3))
        bb_offset = tuple(ovlp.start for ovlp in overlap_bb)

        # load segmentation and affinities for the overlap
        seg = ds_seg[overlap_bb]

        # find the parts off the overlap associated with this block and the neighbor block
        bb_a = tuple(slice(inner_block.begin[i] - off, inner_block.end[i] - off) if i != dim else
                     slice(inner_block.end[i] - off, outer_block.end[i] - off)
                     for i, off in enumerate(bb_offset))

        ngb_block = blocking.getBlockWithHalo(ngb_id, halo)
        inner_ngb, outer_ngb = ngb_block.innerBlock, ngb_block.outerBlock
        bb_b = tuple(slice(inner_ngb.begin[i] - off, inner_ngb.end[i] - off) if i != dim else
                     slice(outer_ngb.begin[i] - off, inner_ngb.begin[i] - off)
                     for i, off in enumerate(bb_offset))

        # add proper offsets to the 2 segmentation halves
        # (but keep mask !)
        bg_mask = seg == 0

        # continue if we have only background in the overlap
        if np.sum(bg_mask) == bg_mask.size:
            continue
        print("Stitching with block", ngb_id)

        affs = ds_affs[(slice(None),) + overlap_bb]
        if affs.dtype == np.dtype('uint8'):
            affs = affs.astype('float32') / 255.
        affs = 1. - affs

        # FIXME this is exactly the opposite of what I have expected ?!
        seg[bb_a] += offsets[ngb_id]
        seg[bb_b] += offsets[block_id]
        seg[bg_mask] = 0

        # TODO restrict merges to segments that actually touch at the overlap surface ??
        # run segmenter to get the node assignment
        n_labels = int(seg.max()) + 1
        merged_nodes = segmenter(seg, affs, n_labels)

        if merged_nodes.size > 0:
            node_assignments.append(merged_nodes)

    print("Finished block", block_id)
    return np.concatenate(node_assignments, axis=0) if node_assignments else None


# TODO we might want to overwrite the non-stitched segmentation later.
def write_stitched_segmentation(block_id, blocking, ds, ds_out, node_labels, offsets):
    print("Write block", block_id)
    off = offsets[block_id]
    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    seg = ds[bb]
    mask = seg != 0
    seg[mask] += off
    seg = nifty.tools.take(node_labels, seg)
    ds_out[bb] = seg


# def write_stitched_segmentation(block_id, blocking, ds, ds_out, merge_nodes, offsets):
#     off = offsets[block_id]
#     block = blocking.getBlock(block_id)
#     bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
#     seg = ds[bb]
#     mask = seg != 0
#     seg[mask] += off
#     un = np.unique(seg)
#     mnodes = np.zeros_like(seg, dtype='uint64')
#     for i, mn in enumerate(merge_nodes):
#         if mn[0] in un:
#             mnodes[seg == mn[0]] = i
#         if mn[1] in un:
#             mnodes[seg == mn[1]] = i
#     ds_out[bb] = mnodes


def stitch_segmentation(block_id, key, segmenter, n_threads=40):

    t0 = time.time()

    offsets = './block_offsets_0%i_%s.json' % (block_id, key)
    # # offsets = './test_offsets_mc_affs_not_stitched.json'
    with open(offsets, 'r') as f:
        offsets = np.array(json.load(f), dtype='uint64')

    # empty_blocks = np.logical_not(offsets)

    # # add up the offsets
    # # ignore blocks which have only background (offset = zero)
    block_list = np.where(offsets > 0)[0]
    empty_blocks = offsets == 0

    last_offset = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    n_labels = offsets[-1] + last_offset + 1

    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/filtered' % block_id
    # path = '/groups/saalfeld/home/papec/data_test.n5'
    f = z5py.File(path)

    chunk_shape = (25, 256, 256)
    # TODO maybe use 1k blocks (factor 4) ?!
    block_shape = tuple(cs * 2 for cs in chunk_shape)
    halo = [5, 50, 50]

    shape = f['gray'].shape
    # shape = (200, 2048, 2048)
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    assert len(empty_blocks) == blocking.numberOfBlocks
    block_list = np.arange(blocking.numberOfBlocks)

    group = 'segmentations'
    in_key = '%s/%s' % (group, key)
    ds = f[in_key]

    ds_affs = f['predictions/full_affs']

    # find the node assignmemnts by running the segmenter on the (segmentation) overlaps

    # results = [stitch_block_neighbors(ds, ds_affs, blocking, block_id,
    #                                   halo, segmenter, offsets, empty_blocks)
    #            for block_id in block_list]

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(stitch_block_neighbors, ds, ds_affs, blocking, block,
                           halo, segmenter, offsets, empty_blocks)
                 for block in block_list]
        results = [t.result() for t in tasks]
    print("Finished stitching")

    results = [res for res in results if res is not None]
    results = np.concatenate(results, axis=0)
    np.save('tmp_%s.npy' % key, results)

    # results = np.load('tmp_%s.npy' % key)
    # assert results.max() < n_labels
    n_labels = int(results.max()) + 1

    # stitch the segmentation (node level)
    ufd = nifty.ufd.ufd(n_labels)
    ufd.merge(results)
    node_labels = ufd.elementLabeling()
    vigra.analysis.relabelConsecutive(node_labels, keep_zeros=True,
                                      start_label=1, out=node_labels)

    # save the stitched result
    out_key = key.split('_')
    out_key.remove('not')
    out_key = '_'.join(out_key)
    out_key = '%s/%s' % (group, out_key)
    if out_key in f:
        ds_out = f[out_key]
    else:
        ds_out = f.create_dataset(out_key, shape=ds.shape,
                                  chunks=chunk_shape, dtype='uint64', compression='gzip')

    # [write_stitched_segmentation(block_id, blocking, ds, ds_out, node_labels, offsets)
    #  for block_id in block_list]

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(write_stitched_segmentation, block, blocking, ds,
                           ds_out, node_labels, offsets)
                 for block in block_list]
        [t.result() for t in tasks]

    t0 = time.time() - t0
    print("Stitching segmentation took", t0, "s")
    return t0


# Stitching times:
# mc-local:
# mc-rf:
# lmc-local:
# lmc-rf:
if __name__ == '__main__':
    block_id = 2
    n_threads = 60
    times = []
    algo = 'mc'
    feat = 'local'
    key2, segmenter = segmenter_factory(algo, feat, return_merged_nodes=True)
    for key1 in ('wslr', 'wsdt'):
        key = '%s_%s' % (key1, key2)
        stitch_time = stitch_segmentation(block_id, key, segmenter, n_threads)
        times.append(stitch_time)
    print(times)
