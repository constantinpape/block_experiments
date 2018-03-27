import numpy as np
import z5py
import nifty
from concurrent import futures


def find_max_id(path, key, n_threads):
    ds = z5py.File(path)[key]
    shape = ds.shape
    block_shape = ds.chunks
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    def max_id_block(block_id):
        print("Find max-id for block", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        return ds[bb].max()

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(max_id_block, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
        max_ids = np.array([t.result() for t in tasks])

    max_id = max_ids.max()
    print("Found max id:", max_id)
    ds.attrs['maxId'] = int(max_id)


if __name__ == '__main__':
    block_id = 2
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    key = 'filtered/segmentations/multicut_more_features'
    find_max_id(path, key, 40)
