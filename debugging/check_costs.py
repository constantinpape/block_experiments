import os
import numpy as np
import z5py
import nifty.graph.rag as nrag


def check_costs(path, key):
    costs = z5py.File(path)[key]

    shape = costs.shape
    chunks = costs.chunks

    print(shape, chunks)
    n_chunks = shape[0] // chunks[0] + 1

    print("Number of chunks:", n_chunks)

    print("Checking chunks for existance ...")
    for chunk_id in range(n_chunks):
        chunk_path = os.path.join(path, key, str(chunk_id))
        if not os.path.exists(chunk_path):
            print("Chunk", chunk_id, "does not exist!")
    print("... done")

    data = costs[:]
    print("Mean costs:", np.mean(data))
    print("Min costs:", np.min(data))
    print("Max costs:", np.max(data))


def compare_cremi_costs():
    path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_A+/tmp_files/costs_pure_affs.n5'
    path_rf = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_A+/tmp_files/costs.n5'

    print("Loading costs...")
    costs = z5py.File(path)['costs'][:]
    print("... done")
    print("Loading rf costs...")
    costs_rf = z5py.File(path_rf)['costs'][:]
    print("... done")

    assert len(costs) == len(costs_rf)

    print("Total nomber of costs", len(costs))

    print("Cost ranges")
    print("Aff-costs:", np.min(costs), np.max(costs), np.mean(costs))
    print("RF-costs: ", np.min(costs_rf), np.max(costs_rf), np.mean(costs_rf))

    # print("Aff-costs:", np.sum(costs == 0))
    print("Rf-costs:", np.sum(costs == 0))

    print("Number of min costs:")
    print("Aff:", np.sum(costs == costs.min()))
    print("Rf :", np.sum(costs_rf == costs_rf.min()))

    print("Number of max costs:")
    print("Aff:", np.sum(costs == costs.max()))
    print("Rf :", np.sum(costs_rf == costs_rf.max()))


def check_features():
    path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_A+/tmp_files/features.n5'
    feats = z5py.File(path)['features'][:]
    print(feats.shape)
    for ii in range(feats.shape[1]):
        print("Feature index", ii)
        print(np.sum(feats[:, ii] == 0))
        print(np.mean(feats[:, ii]), '+-', np.std(feats[:, ii]))
        print(np.max(feats[:, ii]), np.min(feats[:, ii]))


def compute_rag_feats():
    path = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi_warped/sampleA+.n5'
    print("Loading seg")
    seg = z5py.File(path)['segmentations/watershed'][:]
    print(seg.shape)
    print("Computing rag")
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max() + 1))
    print("Loading affs")
    affs = z5py.File(path)['predictions/affs_glia'][0:12, :]
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    assert len(offsets) == len(affs)
    print("Computing feats")
    feats = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets)

    f_save = z5py.File('./costs_tmp.n5')
    ds = f_save.create_dataset('feats', dtype='float32', shape=feats.shape,
                               chunks=(feats.shape[0], 1), compression='gzip')
    ds[:] = feats


if __name__ == '__main__':
    # path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/scotts_block_843c5df9fbef6bd70ed6b48c9d0c6631/tmp_files/costs.n5'
    # path = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/scotts_block_e4fc96b3af2e6eb9fa92d9d6745b804c/tmp_files/costs.n5'
    # key = 'costs'
    # check_costs(path, key)

    # compare_cremi_costs()
    # check_features()
    compute_rag_feats()
