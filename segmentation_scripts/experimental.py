import z5py
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view


def compare_lr_ws():
    # f = z5py.File('/home/papec/mnt/papec/Work/neurodata_hdd/scotts_blocks/data_test.n5')
    f = z5py.File('/home/papec/Work/neurodata_hdd/scotts_blocks/data_test_small.n5')
    affs = 1. - f['predictions/full_affs'][:3, :]

    # wslr1 = cseg.LRAffinityWatershed(0.1, 0.2, 1.6)
    wslr2 = cseg.LRAffinityWatershed(0.01, 0.2, 1.6, return_seeds=True)

    # ws1, _ = wslr1(affs)
    ws2, _, seeds = wslr2(affs)

    raw = f['gray'][:]
    view([raw, affs.transpose((1, 2, 3, 0)), ws2, seeds],
         ['raw', 'affs', 'ws-lrz', 'seeds'])


if __name__ == '__main__':
    compare_lr_ws()
