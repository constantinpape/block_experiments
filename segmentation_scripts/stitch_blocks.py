import os
import numpy as np

import nifty
import z5py
from blockwise_segmentations import segmenter_factory


def stitch_block_neighbors(ds_seg, ds_affs, blocking, block_id, halo, segmenter):

    block = blocking.getBlockWithHalo(block_id, halo)
    inner_block, outer_block = block.innerBlock, block.outerBlock

    # iterate over the (upper) neighbors of this block and stitch with the segmenter
    to_lower = False
    for dim in range(3):
        ngb_id = blocking.getNeighborId(block_id, dim, to_lower)
        if ngb_id == -1:
            continue
        # ngb_block = blocking.getBlockWithHalo(ngb_id, halo)
        # find the overlap
        overlap_bb = tuple(slice(inner_block.begin[i], inner_block.end[i]) if i != dim else
                           slice(inner_block.end[i] - halo[i], outer_block.end[i])
                           for i in range(3))
        # load offsets and affinities for the overlap
        seg = ds_seg[overlap_bb]
        affs = ds_affs[(slice(None),) + overlap_bb]
        # TODO find 2 segmentation halves
        # TODO add proper offsets to the 2 segmentation halves
        # TODO run segmenter and find the corresponding node assignments


# TODO offsets
def stitch_segmentation(block_id, key, segmenter):
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

    # TODO also need to pass the affinity dataset as an argument
    # TODO parallelize
    results = [stitch_block_neighbors(ds, blocking, block_id, halo, segmenter)
               for block_id in range(blocking.numberOfBlocks)]

    # TODO save the stitched result
    out_key = key.split('_')
    out_key.remove('not')
    out_key = '_'.join(out_key)
    out_key = '%s/%s' % (group, out_key)

    # ds_out = f.create_dataset(out_key, shape=)


if __name__ == '__main__':
    block_id = 2
    key, segmenter = segmenter_factory()
    stitch_segmentation(block_id, key, segmenter)
