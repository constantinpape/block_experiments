import json
import time
from concurrent import futures

# the good old anti-segfault vigra import ...
import vigra
import numpy as np
import nifty
import z5py

from backend import segmenter_factory, fragmenter_factory


# TODO return the max id of the multicut to find proper offsets
def segment_block(path, out_key, blocking, block_id,
                  halo, fragmenter, segmenter):
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
                     for start, stop in zip(block.innerBlockLocal.begin,
                                            block.innerBlockLocal.end))

    # load mask
    # mask = f['masks/combined_mask'][bb].astype('bool')
    mask = f['masks/min_filter_mask'][bb].astype('bool')
    # if we only have mask, continue
    if np.sum(mask) == 0:
        print("Finished", block_id, "; contained only mask")
        # return False, time.time() - t0
        return 0, time.time() - t0

    # load affinities and convert them to proper format
    affs = f['predictions/full_affs'][(slice(None),) + bb]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    ws, max_id = fragmenter(affs, mask)
    segmentation = segmenter(ws, affs, max_id + 1)
    ds_out = f[out_key]

    segmentation = segmentation[local_bb].astype('uint64')
    segmentation, max_id, _ = vigra.analysis.relabelConsecutive(segmentation, start_label=1)
    mask = mask[local_bb]
    ignore_mask = np.logical_not(mask)
    segmentation[ignore_mask] = 0

    ds_out[inner_bb] = segmentation
    print("Finished", block_id)
    return max_id, time.time() - t0


def run_segmentation(block_id, fragmenter, segmenter, key, n_threads=60):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/filtered' % block_id
    # path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/scotts_blocks/data_test.n5'
    # path = '/home/papec/Work/neurodata_hdd/scotts_blocks/data_test_large.n5'
    f = z5py.File(path)
    shape = f['gray'].shape

    chunk_shape = (25, 256, 256)
    # TODO maybe use 1k blocks (factor 4) ?!
    block_shape = tuple(cs * 2 for cs in chunk_shape)
    halo = [5, 50, 50]

    # shape = (200, 2048, 2048)
    # central = tuple(sh // 2 for sh in shape)
    # offset = (100, 1000, 1000)
    # bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))

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
        f.attrs['blocking'] = block_shape

    t0 = time.time()
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(segment_block, path, out_key,
                           blocking, block, halo,
                           fragmenter, segmenter)
                 for block in range(blocking.numberOfBlocks)]
        results = [t.result() for t in tasks]
    # block_times = [segment_block(path, out_key, blocking, block_id, halo, segmenter)
    #                for block_id in [34]]  # range(blocking.numberOfBlocks)]
    t_tot = time.time() - t0
    print("Total processing time:", t_tot)

    times = {'blocks': [res[1] for res in results], 'total': t_tot}
    with open('./timings_0%i_%s.json' % (block_id, key), 'w') as ft:
        json.dump(times, ft)

    offsets = [res[0] for res in results]
    with open('./block_offsets_0%i_%s.json' % (block_id, key), 'w') as ft:
        json.dump(offsets, ft)


if __name__ == '__main__':
    block_id = 2
    algo = 'mc'
    feat = 'local'
    n_threads = 40
    key2, segmenter = segmenter_factory(algo, feat)
    for ws in ('wslr', 'wsdt', 'mws', 'wsdt_pre'):
        key1, fragmenter = fragmenter_factory(ws)
        key = '%s_%s' % (key1, key2)
        run_segmentation(block_id, fragmenter, segmenter, key, n_threads)
