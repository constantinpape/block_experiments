import os
from concurrent import futures
from skimage.draw import circle

import z5py
import nifty
import numpy as np
from cremi_tools.skeletons import build_skeleton_metrics


# TODO may want to use the implementation to cremi_tools
def visualize_skeletons(lauritzen_block_id, seg_key, out_key, n_threads, radius=10):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % lauritzen_block_id
    key1 = '/'.join(('filtered', 'segmentations', seg_key))
    label_file = os.path.join(path, key1)
    skeleton_file = os.path.join(path, 'skeletons')

    # FIXME this is pretty inefficient, because we repeat the same computation twice,
    # but for now these things are fast enough
    metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)
    non_empty_chunks, skeletons_to_blocks = metrics.groupSkeletonBlocks(n_threads)

    f = z5py.File(path)
    shape = f[key1].shape
    chunks = f[key1].chunks

    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(chunks))

    if out_key not in f:
        ds = f.create_dataset(out_key, dtype='uint64', compression='gzip', shape=shape, chunks=chunks)
    else:
        ds = f[out_key]

    def visualize_block(block_id):
        print("Writing block", block_id)
        block = blocking.getBlock(block_id)
        bshape = tuple(block.shape)
        vol = np.zeros(bshape, dtype='uint64')
        skeletons = skeletons_to_blocks[block_id]

        for skel_id, coords in skeletons.items():
            for coord in coords:
                _, z, y, x = coord
                rr, cc = circle(y, x, radius, shape=bshape[1:])
                vol[z, rr, cc] = skel_id

        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        ds[bb] = vol

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(visualize_block, block_id) for block_id in non_empty_chunks]
        [t.result() for t in tasks]


if __name__ == '__main__':
    block_id = 2
    visualize_skeletons(block_id, 'multicut_more_features', 'skeletons/volume', 40)
