import z5py
import nifty
import numpy as np
import cremi_tools.segmentation as cseg
from cremi_tools.viewer.volumina import view
import nifty.graph.rag as nrag


def compare_lr_ws():
    # f = z5py.File('/home/papec/mnt/papec/Work/neurodata_hdd/scotts_blocks/data_test.n5')
    f = z5py.File('/home/papec/Work/neurodata_hdd/scotts_blocks/data_test_small.n5')
    affs = 1. - f['predictions/full_affs'][:3, :]

    wslr1 = cseg.LRAffinityWatershed(0.1, 0.2, 1.6)
    wslr2 = cseg.LRAffinityWatershed(0.01, 0.2, 1.6, return_seeds=True)

    ws1, _ = wslr1(affs)
    ws2, _, seeds = wslr2(affs)

    raw = f['gray'][:]
    view([raw, affs.transpose((1, 2, 3, 0)), ws1, ws2, seeds],
         ['raw', 'affs', 'ws', 'ws-lrz', 'seeds'])


def agglomerate_wsdt(thresh=.1, size_thresh=500):
    f = z5py.File('/home/papec/Work/neurodata_hdd/scotts_blocks/data_test_small.n5')
    affs = 1. - f['predictions/full_affs'][:3, :]
    affs_xy = np.mean(affs[1:3], axis=0)
    affs_z = affs[0]

    wsdt = cseg.DTWatershed(0.2, 1.6)
    ws, max_id = wsdt(affs_xy)
    rag = nrag.gridRagStacked2D(ws.astype('uint32'),
                                numberOfLabels=int(max_id + 1),
                                dtype='uint32')
    features_z = nrag.accumulateEdgeStandardFeatures(rag, affs_z, keepZOnly=True, zDirection=2)[1]
    features_z = features_z[:, 0]
    edge_offset = rag.totalNumberOfInSliceEdges
    edge_sizes = rag.edgeLengths()[edge_offset:]

    uvs = rag.uvIds()[edge_offset:]
    assert len(features_z) == len(uvs)
    # TODO filter by edge overlap as well !
    merge_edges = np.logical_and(features_z < thresh, edge_sizes > size_thresh)
    merge_nodes = uvs[merge_edges]

    ufd = nifty.ufd.ufd(rag.numberOfNodes)
    ufd.merge(merge_nodes)
    node_labels = ufd.elementLabeling()
    ws_merged = nrag.projectScalarNodeDataToPixels(rag, node_labels)

    raw = f['gray'][:]
    view([raw, affs.transpose((1, 2, 3, 0)), ws, ws_merged],
         ['raw', 'affs', 'ws', 'ws-merged'])


if __name__ == '__main__':
    agglomerate_wsdt()
    # compare_lr_ws()
