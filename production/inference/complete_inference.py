from __future__ import print_function
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call

import z5py
from simpleference.inference.util import offset_list_from_precomputed
from precompute_offsets import precompute_offset_list

networks = {'mala': {'prefix': 'unet_mala',
                     'input_shape': (88, 808, 808),
                     'output_shape': (60, 596, 596)}}


def single_inference(path, network_key, gpu, iteration):
    call(['./run_inference.sh', path, network_key, str(gpu), str(iteration)])


def complete_inference(path, network_key, gpu_list, iteration):

    assert os.path.exists(path), "Path to N5 dataset with raw data and mask does not exist"
    f = z5py.File(path, use_zarr_format=False)
    assert 'gray' in f, "Raw data not present in N5 dataset"
    assert 'masks/initial_mask' in f, "Mask not present in N5 dataset"

    shape = f['gray'].shape

    output_shape = networks[network_key]['output_shape']

    # create the datasets
    # the n5 datasets might exist already
    target_key = 'predictions/affs_glia'
    if target_key not in f:

        if 'predictions' not in f:
            f.create_group('predictions')

        if output_shape[0] > 30 and all(outs % 2 == 0 for outs in output_shape):
            chunks = (3,) + tuple(outs // 2 for outs in output_shape)
        else:
            chunks = (3,) + output_shape

        aff_shape = (13,) + shape
        f.create_dataset(target_key,
                         shape=aff_shape,
                         compression='gzip',
                         dtype='uint8',
                         chunks=chunks)

    # make the offset files, that assign blocks to gpus
    # generate offset lists with mask
    offset_list = precompute_offset_list(path, output_shape)
    offset_list_from_precomputed(offset_list, gpu_list, './offsets')

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, path, network_key, gpu, iteration)
                 for gpu in gpu_list]
        [t.result() for t in tasks]


if __name__ == '__main__':
    gpu_list = list(range(8))
    iteration = 140000
    network_key = 'mala'
    for block_id in (3, 4):
        path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/raw' % block_id
        complete_inference(path, network_key, gpu_list, iteration)
