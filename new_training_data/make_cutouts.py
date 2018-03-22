import z5py
from cremi_tools.viewer.volumina import view


locations = [[284, 2075, 3446],
             [436, 1965, 2260],
             [324, 2670, 3068],
             [179, 5450, 1222]]


# TODO raw or filtered grayscale ?!?
def make_cutout(location, size):
    path = '/home/papec/mnt/nrs/lauritzen/02/workspace.n5'
    key_raw = 'filtered/gray'
    key_affs = 'filtered/predictions/full_affs'
    key_ws = 'filtered/segmentations/watershed_2d'
    key_seg = 'filtered/segmentations/multicut_more_features'

    f = z5py.File(path)

    bb = tuple(slice(loc - si, loc + si) for loc, si in zip(location, size))
    raw = f[key_raw][bb]
    ws = f[key_ws][bb]
    seg = f[key_seg][bb]
    affs = f[key_affs][(slice(None),) + bb]

    view([raw, ws, seg], ['raw', 'wsdt', 'mc'])


if __name__ == '__main__':
    size = (50, 512, 512)
    make_cutout(locations[3], size)
