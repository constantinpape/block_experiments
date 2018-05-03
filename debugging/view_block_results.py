import json
import numpy as np
import nifty
import z5py
from concurrent import futures
from cremi_tools.viewer.volumina import view


def view_block(lauritzen_block_id, block_id, block_shape=(52, 512, 512), halo=[0, 0, 0]):
    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/raw' % lauritzen_block_id
    f = z5py.File(path)

    ds_raw = f['gray']
    shape = ds_raw.shape

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))
    if sum(halo) > 0:
        block = blocking.getBlockWithHalo(block_id, halo).outerBlock
    else:
        block = blocking.getBlock(block_id)

    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

    raw = ds_raw[bb]
    seg = f['segmentations/mc_glia_affs'][bb]

    mask1 = f['masks/initial_mask'][bb]
    mask2 = f['masks/minfilter_mask'][bb]

    view([raw, seg, mask1.astype('uint32'), mask2.astype('uint32')],
         ['raw', 'seg', 'initial mask', 'minfilter mask'])


def find_border_blocks(block_id, block_shape=(52, 512, 512)):
    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/raw' % block_id
    f = z5py.File(path)

    ds_raw = f['gray']
    shape = ds_raw.shape

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))
    mask = f['masks/minfilter_mask']

    def check_if_border(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data = mask[bb]
        ration_in_mask = np.sum(data) / data.size

        if 0.3 < ration_in_mask < 0.7:
            return block_id
        else:
            return None

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(find_border_blocks, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
        block_list = [t.result() for t in tasks]
    block_list = [block for block in block_list if block is not None]

    with open('border_blocks_%i.json' % block_id, 'w') as f:
        json.dump(block_list, f)


if __name__ == '__main__':
    find_border_blocks(2)
    # view_block(1, 159, halo=[10, 200, 200])
