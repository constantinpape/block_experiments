import sys
import numpy as np
# import vigra
sys.path.append('/home/papec/Work/my_projects/cremi_tools')
sys.path.append('/home/papec/Work/my_projects/z5/bld/python')
import z5py
from cremi_tools.viewer.volumina import view


def view_segmentations(block_id, keys):

    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5' % block_id
    f = z5py.File(path)
    ds = f['filtered/gray']

    # shape = ds.shape
    # central = tuple(sh // 2 for sh in shape)
    # offset = (50, 500, 500)
    # offset = (100, 1000, 1000)
    # bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))
    bb = np.s_[:200, :2048, :2048]

    print("Loading raw from", bb)
    data = [ds[bb]]
    labels = ['raw']

    print("Loading segmentations ...")
    data.extend([f['filtered/%s' % key][bb] for key in keys])
    labels.extend(keys)

    # data.extend([f['filtered/masks/combined_mask'][bb] for key in keys])
    # labels.extend(keys)

    # mask_a = f['filtered/masks/min_filter_mask'][bb]
    # mask_b = f['filtered/masks/cortex_mask'][bb].astype('bool')
    # mask_c = mask_a.copy()
    # mask_c[mask_b] = 0
    # data.append(mask_a.astype('uint32'))
    # data.append(mask_b.astype('uint32'))
    # data.append(mask_c.astype('uint32'))
    # labels.append('minfilter_mask')
    # labels.append('cortex_mask')
    # labels.append('combined_mask')

    view(data, labels)


def extract_data(block_id):
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


def check_cremi():
    folder = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi_warped/sampleA+.n5'
    bb = np.s_[50:100, 512:1024, 512:1024]
    f = z5py.File(folder)
    mc1 = f['segmentations/multicut_rf'][bb]
    mc2 = f['segmentations/multicut'][bb]
    ws = f['segmentations/watershed'][bb]
    raw = f['raw'][bb]
    view([raw, ws, mc1, mc2])


if __name__ == '__main__':
    # check_cremi()
    # extract_data(2)
    keys = ['segmentations/watershed',
            'segmentations/watershed_2d']
    view_segmentations(2, keys)
