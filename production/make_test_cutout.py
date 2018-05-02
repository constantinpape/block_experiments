import numpy as np
import z5py


def make_test_cutout(block_id, out_path, bb):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/raw' % block_id
    f = z5py.File(path)

    out = z5py.File(out_path)

    raw = f['gray'][bb]
    ds = out.create_dataset('gray', shape=raw.shape, chunks=(26, 256, 256),
                            compression='gzip', dtype='uint8')
    ds[:] = raw

    mask = f['masks/min_filter_mask'][bb]
    out.create_group('masks')
    ds = out.create_dataset('masks/min_filter_mask', shape=mask.shape, chunks=(26, 256, 256),
                            compression='gzip', dtype='uint8')
    ds[:] = mask

    affs = f['predictions/affs_glia'][(slice(None),) + bb]
    out.create_group('predictions')
    ds = out.create_dataset('predictions/affs_glia', shape=affs.shape, chunks=(3, 26, 256, 256),
                            compression='gzip', dtype='uint8')
    ds[:] = affs


if __name__ == '__main__':
    bb = np.s_[100:200, 1000:2024, 1000:2024]
    out_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/lauritzen/production_test_block.n5'
    make_test_cutout(3, out_path, bb)
