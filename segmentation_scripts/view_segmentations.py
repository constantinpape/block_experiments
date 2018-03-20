import sys
import numpy as np
import vigra
sys.path.append('/home/papec/Work/my_projects/cremi_tools')
sys.path.append('/home/papec/Work/my_projects/z5/bld/python')


def view_segmentations(block_id, keys):
    import z5py
    from cremi_tools.viewer.volumina import view

    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5' % block_id
    f = z5py.File(path)
    ds = f['filtered/gray']

    shape = ds.shape
    central = tuple(sh // 2 for sh in shape)
    offset = (100, 1000, 1000)
    bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))

    print("Loading raw from", bb)
    data = [ds[bb]]
    labels = ['raw']

    print("Loading segmentations ...")
    data.extend([f['filtered/segmentations/%s' % key][bb] for key in keys])
    labels.extend(keys)

    view(data, labels)


def extract_data(block_id):
    import z5py
    # path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5' % block_id
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    f = z5py.File(path)
    f_out = z5py.File('./data_test_large.n5')
    ds = f['filtered/gray']

    shape = ds.shape
    central = tuple(sh // 2 for sh in shape)
    offset = (100, 1000, 1000)
    bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))
    # bb = np.s_[:200, :1536, :1536]

    raw = ds[bb]
    dsr = f_out.create_dataset('raw', shape=raw.shape, dtype='uint8',
                               compression='gzip', chunks=ds.chunks)
    dsr[:] = raw

    # # vigra.writeHDF5(raw, 'data_test_large.h5', 'raw', compression='gzip')

    ds_affs = f['filtered/predictions/full_affs']
    affs = ds_affs[(slice(None),) + bb]
    dsa = f_out.create_dataset('affs', shape=affs.shape, dtype='float32',
                               compression='gzip', chunks=ds_affs.chunks)
    dsa[:] = affs
    # vigra.writeHDF5(affs, 'data_test_large.h5', 'affs', compression='gzip')

    ds_mask = f['filtered/masks/min_filter_mask']
    mask = ds_mask[bb]
    dsm = f_out.create_dataset('mask', shape=mask.shape, dtype='uint8',
                               compression='gzip', chunks=ds.chunks)
    dsm[:] = mask
    # vigra.writeHDF5(affs, 'data_test_large.h5', 'affs', compression='gzip')


if __name__ == '__main__':
    extract_data(2)
    # keys = ['mc_affs_not_stitched', 'mc_rf_not_stitched']
    # view_segmentations(2, keys)
