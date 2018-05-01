#! /usr/bin/python

import argparse
import os
import json
import nifty
import z5py


def blocks_to_jobs(cache_folder, n_jobs, shape, block_shape):
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)
    block_list = list(range(n_blocks))
    for job_id in range(n_jobs):
        out_file = os.path.join(cache_folder, 'block_list_job%i.json' % job_id)
        with open(out_file, 'w') as f:
            json.dump(block_list[job_id::n_jobs], f)


def create_volumes(f, out_key, shape, block_shape):
    if 'segmentations' not in f:
        f.create_group('segmentations')

    if block_shape[0] > 26:
        chunks = tuple(bs // 2 for bs in block_shape)
    else:
        chunks = block_shape

    if out_key not in f:
        f.create_dataset(out_key, shape=shape, dtype='uint64', compression='gzip', chunks=chunks)


def prepare(path, out_key, cache_folder, n_jobs, block_shape):
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    f = z5py.File(path)
    shape = f['gray'].shape

    create_volumes(f, out_key, shape, block_shape)
    blocks_to_jobs(cache_folder, n_jobs, shape, block_shape)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--block_shape', type=int, nargs=3)

    args = parser.parse_args()
    prepare(args.path, args.out_key,
            args.cache_folder, args.n_jobs,
            tuple(args.block_shape))
