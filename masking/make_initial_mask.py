import z5py
from concurrent import futures
from scipy.ndimage.morphology import binary_closing
# from cremi_tools.viewer.volumina import view


def make_mask(block_id):
    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/raw' % block_id

    f = z5py.File(path)
    ds = f['gray']
    ds_mask = f.create_dataset('masks/initial_masks_', shape=ds.shape, chunks=(26, 256, 256),
                               compression='gzip', dtype='uint8')

    def mask_slice(z):
        print("Start slice", z, "/", ds.shape[0])
        data = ds[z:z+1].squeeze()
        maskz = data > 0
        maskz = binary_closing(maskz, iterations=6)
        ds_mask[z:z+1] = maskz[None]
        print("Done slice", z, "/", ds.shape[0])

    with futures.ThreadPoolExecutor(max_workers=8) as tp:
        tasks = [tp.submit(mask_slice, z) for z in range(ds.shape[0])]
        [t.result() for t in tasks]


if __name__ == '__main__':
    make_mask(1)
