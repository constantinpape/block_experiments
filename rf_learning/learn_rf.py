import pickle
import numpy as np
import nifty.graph.rag as nrag
import z5py
import vigra
from sklearn.ensemble import RandomForestClassifier


# TODO
def affinity_features(rag, affs):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets)
    return np.nan_to_num(features)


def glia_features(rag, seg, glia):
    uv_ids = rag.uvIds()
    statistics = ["Mean", "Count", "Kurtosis", "Maximum",
                  "Minimum", "Quantiles", "RegionRadii",
                  "Skewness", "Sum", "Variance"]
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
    uv_ids = rag.uv_ids()
    node_labels = nrag.gridRagAccumulateLabels(rag, gt)
    edge_labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype('uint8')
    edge_mask = (node_labels[uv_ids] != 0).all(axis=1)
    return edge_labels, edge_mask


def learn_rf(paths, save_path):
    all_features = []
    all_labels = []
    for path in paths:
        f = z5py.File(path)
        seg = f['volumes/labels/watershed'][:]
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

        affs = f['volumes/predictions/affinities'][:]
        if affs.dtype == np.dtype('uint8'):
            affs = affs.astype('float32')
            affs /= 255.
        glia = affs[-1:]
        affs = 1. - affs[:-1]

        features = np.concatenate([affinity_features(rag, affs),
                                   glia_features(rag, seg, glia)],
                                  axis=1)

        gt = f['volumes/labels/neuron_ids'][:]
        labels, edge_mask = edge_labels(rag, gt)
        assert features.shape[0] == labels.shape[0]

        all_features.append(features[edge_mask])
        all_labels.append(labels[edge_mask])

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    assert all_features.shape[0] == all_labels.shape[0]

    rf = RandomForestClassifier(n_jobs=40, n_estimators=200)
    rf.fit(all_features, all_labels)
    with open(save_path, 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    save_path = ''
    samples = ['0', '1', '2', 'A', 'B', 'C']
    # TODO
    paths = ['%s.n5' % sample
             for sample in samples]
    learn_rf(samples, save_path)
