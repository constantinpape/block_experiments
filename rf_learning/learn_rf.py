import os
import pickle
import numpy as np
import nifty.graph.rag as nrag
import z5py
import vigra
from sklearn.ensemble import RandomForestClassifier


FEATURES = ('default_affs',
            'separate_affs',
            'glia')


def affinity_features(rag, affs):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets)
    return np.nan_to_num(features)


def separate_channel_features(rag, affs):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    assert len(affs) == len(offsets)
    features = []
    for channel_id in range(len(affs)):
        features.append(nrag.accumulateAffinityStandartFeatures(rag,
                                                                affs[channel_id:channel_id+1],
                                                                [offsets[channel_id]]))
    features = np.concatenate(features, axis=1)
    return np.nan_to_num(features)


def glia_features(rag, seg, glia):
    uv_ids = rag.uvIds()
    # FIXME for some reason 'Quantiles' are not working
    statistics = ["Mean", "Variance", "Skewness", "Kurtosis",
                  "Minimum", "Maximum", "Count", "RegionRadii"]
    extractor = vigra.analysis.extractRegionFeatures(glia, seg.astype('uint32'),
                                                     features=statistics)

    node_features = np.concatenate([extractor[stat_name][:, None].astype('float32')
                                    if extractor[stat_name].ndim == 1
                                    else extractor[stat_name].astype('float32')
                                    for stat_name in statistics],
                                   axis=1)
    fU = node_features[uv_ids[:, 0], :]
    fV = node_features[uv_ids[:, 1], :]

    edge_features = [np.minimum(fU, fV),
                     np.maximum(fU, fV),
                     np.abs(fU - fV),
                     fU + fV]
    edge_features = np.concatenate(edge_features, axis=1)
    return np.nan_to_num(edge_features)


def edge_labels(rag, gt):
    uv_ids = rag.uvIds()
    node_labels = nrag.gridRagAccumulateLabels(rag, gt)
    edge_labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype('uint8')
    edge_mask = (node_labels[uv_ids] != 0).all(axis=1)
    print(np.sum(edge_mask), "edges of", len(uv_ids), "are valid")
    return edge_labels, edge_mask


def learn_rf(paths, save_path, feature_keys):
    assert feature_keys
    assert all(feat in FEATURES for feat in feature_keys)
    all_features = []
    all_labels = []
    for path in paths:
        print("Computing features and labels for", path)
        f = z5py.File(path)
        seg = f['volumes/labels/watershed'][:]
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

        affs = f['volumes/predictions/affinities'][:]
        if affs.dtype == np.dtype('uint8'):
            print("Cast affinities to float")
            affs = affs.astype('float32')
            affs /= 255.
        glia = affs[-1]
        # affs = 1. - affs[:-1]
        affs = affs[:-1]
        print(affs.min(), affs.max())

        features = []
        if 'default_affs' in feature_keys:
            features.append(affinity_features(rag, affs))
        if 'separate_affs' in feature_keys:
            features.append(separate_channel_features(rag, affs))
        if 'glia' in feature_keys:
            features.append(glia_features(rag, seg, glia))

        features = np.concatenate(features, axis=1)

        mask = f['volumes/labels/mask'][:]
        gt = f['volumes/labels/neuron_ids'][:]
        gt[np.logical_not(mask)] = 0
        labels, edge_mask = edge_labels(rag, gt)
        assert features.shape[0] == labels.shape[0]

        all_features.append(features[edge_mask])
        all_labels.append(labels[edge_mask])

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    assert all_features.shape[0] == all_labels.shape[0]

    rf = RandomForestClassifier(n_jobs=40, n_estimators=200)
    rf.fit(all_features, all_labels)
    rf.n_jobs = 1
    with open(save_path, 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    train_folder = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419'

    features = ['default_affs']
    features.sort()

    feat_str = '_'.join(features)
    save_path = os.path.join(train_folder, 'rf_%s.pkl' % feat_str)

    samples = ['0', '1', '2_', 'A', 'B', 'C_']
    paths = [os.path.join(train_folder, 'n5', 'sample%s.n5' % sample)
             for sample in samples]

    learn_rf(paths, save_path, features)
