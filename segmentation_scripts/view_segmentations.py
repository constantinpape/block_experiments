import sys
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


if __name__ == '__main__':
    keys = ['mc_affs_not_stitched', 'mc_rf_not_stitched']
    view_segmentations(2, keys)
