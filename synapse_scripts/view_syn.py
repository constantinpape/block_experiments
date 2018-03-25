import numpy as np
import z5py
from cremi_tools.viewer.volumina import view


def view_synapses(bb):
    raw_path = '/nrs/saalfeld/lauritzen/02/workspace.n5/filtered'
    raw_key = 'gray'
    syn_path = '/nrs/saalfeld/lauritzen/02/workspace.n5'
    syn_key = 'syncleft_dist_DTU-2_200000'

    raw = z5py.File(raw_path)[raw_key][bb]
    syn = z5py.File(syn_path)[syn_key][bb]

    view([raw, syn])


if __name__ == '__main__':
    bb = np.s_[1000:1100, 2000:2512, 2000:2512]
    view_synapses(bb)
