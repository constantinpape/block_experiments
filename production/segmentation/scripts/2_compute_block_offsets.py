#! /usr/bin/python

import argparse
import os
import json
import numpy as np


def compute_block_offsets(cache_folder, n_jobs):
    offset_dict = {}
    for job_id in range(n_jobs):
        offset_file = os.path.join(cache_folder, 'block_offsets_job%i.json' % job_id)
        with open(offset_file) as f:
            offset_dict.update(json.load(f))

    block_ids = list(offset_dict.keys())
    block_ids.sort()
    offsets = np.array([offset_dict[block_id] for block_id in block_ids], dtype='uint64')

    empty_blocks = np.where(offsets == 0)[0]

    last_offset = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    n_labels = int(offsets[-1] + last_offset + 1)

    out_file = os.path.join(cache_folder, './block_offsets.json')
    with open(out_file, 'w') as f:
        out_dict = {'offsets': offsets.tolist(),
                    'empty_blocks': empty_blocks.tolist(),
                    'n_labels': n_labels}
        json.dump(out_dict, f)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    compute_block_offsets(args.cache_folder, args.n_jobs)
