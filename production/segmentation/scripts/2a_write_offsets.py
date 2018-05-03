#! /usr/bin/python

import os
import json
import argparse
import z5py
import numpy as np
import nifty


def write_block(block_id, blocking, ds, offsets):
    off = offsets[block_id]

    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    seg = ds[bb]

    mask = seg != 0

    # don't write empty blocks
    if np.sum(mask) == 0:
        # print("skipping write", block_id)
        return

    seg[mask] += off
    ds[bb] = seg


def write_offsets(path, out_key, cache_folder, job_id, block_shape):

    offsets_path = os.path.join(cache_folder, 'block_offsets.json')
    with open(offsets_path) as f:
        offset_dict = json.load(f)
        offsets = offset_dict['offsets']
        max_label = offset_dict['n_labels'] - 1

    input_file = os.path.join(cache_folder, 'block_list_job%i.json' % job_id)
    with open(input_file) as f:
        block_list = json.load(f)

    shape = z5py.File(path)['gray'].shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    f = z5py.File(path)
    ds = f[out_key]

    [write_block(block_id, blocking, ds, offsets)
     for block_id in block_list]

    # write out the max-label with job 0
    if job_id == 0:
        ds.attrs['maxId'] = max_label

    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('--block_shape', type=int, nargs=3)

    args = parser.parse_args()
    write_offsets(args.path, args.out_key,
                  args.cache_folder, args.job_id,
                  tuple(args.block_shape))
