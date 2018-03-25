import numpy as np
import z5py
from cremi_tools.viewer.volumina import view


def view_synapses(bb):
    raw_path = '/home/papec/mnt/nrs/lauritzen/02/workspace.n5/filtered'
    raw_key = 'gray'
    syn_path = '/home/papec/mnt/nrs/lauritzen/02/workspace.n5'
    syn_key = 'syncleft_dist_DTU-2_200000'

    raw = z5py.File(raw_path)[raw_key][bb]
    syn = z5py.File(syn_path)[syn_key][bb]

    syn -= syn.min()
    syn /= syn.max()

    thresh = (syn > .5).astype('uint32')
    print(syn.min(), syn.max())

    view([raw, syn, thresh])


if __name__ == '__main__':
    bb = np.s_[400:450, 2000:2512, 2000:2512]
    view_synapses(bb)
