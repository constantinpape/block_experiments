# import nifty
import z5py

from cremi_tools import view


def view_subres(central_location, halo):
    path = '/nrs/saalfeld/sample_E/sample_E.n5'
    path2 = '/groups/saalfeld/saalfeldlab/sampleE'
    f = z5py.File(path)
    f2 = z5py.File(path2)

    ds_raw = f['volumes/raw/s0']
    ds_affs = f['volumes/predictions/affs_glia']
    ds_ws = f2['segmentations/watershed_glia']

    bb = tuple(slice(cl - ha, cl + ha)
               for cl, ha in zip(central_location, halo))
    bb_affs = (slice(0, 3),) + bb

    # TODO multi-threaded read ?!
    raw = ds_raw[bb]
    affs = ds_affs[bb_affs]
    ws = ds_ws[bb]

    view([raw, affs, ws])


def dataset_middle():
    path = '/nrs/saalfeld/sample_E/sample_E.n5'
    f = z5py.File(path)
    ds_raw = f['volumes/raw/s0']
    shape = ds_raw
    return tuple(sh // 2 for sh in shape)


if __name__ == '__main__':
    halo = (50, 512, 512)
    central_location = dataset_middle()
    view_subres(central_location, halo)
