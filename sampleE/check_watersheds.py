import json
import nifty
import z5py
from concurrent import futures


def check_watershed(path, ws_key, mask_key, block_shape=(50, 512, 512), n_threads=70):
    f = z5py.File(path)
    ds_ws = f[ws_key]
    ds_mask = f[mask_key]

    shape = ds_ws.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    def check_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end)
                   for beg, end in zip(block.begin, block.end))
        ws = ds_ws[bb]
        mask = ds_mask[bb].astype('bool')
        ws_mask = ws != 0

        if mask.sum() != ws_mask.sum():
            return block_id
        else:
            return None

    print("Checking watersheds...")
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(check_block, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
        results = [t.result() for t in tasks]

    results = [res for res in results if res is not None]

    with open('ws_checks.json', 'w') as f:
        json.dump(results, f)

    for block_id in results:
        print("Ws and mask do not agree for", block_id)


if __name__ == '__main__':
    path = '/groups/saalfeld/saalfeldlab/sampleE'
    ws_key = 'segmentations/watershed_glia'
    mask_key = 'masks/minfilter_mask'
    check_watershed(path, ws_key, mask_key)
