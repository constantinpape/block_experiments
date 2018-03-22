import os
import pickle
from functools import partial

import numpy as np
import nifty
import nifty.graph.rag as nrag

import cremi_tools.segmentation as cseg


#
# over-segmentation with watershed based fragmenters
#

def long_range_ws(affs, mask, thresh=0.01):
    wsfu = cseg.LRAffinityWatershed(threshold_cc=thresh, threshold_dt=0.2, sigma_seeds=2.,
                                    size_filter=50)
    return wsfu(affs, mask)


def wsdt(affs, mask):
    ws_input = np.mean(affs[1:3], axis=0)
    wsfu = cseg.DTWatershed(threshold_dt=0.2, sigma_seeds=1.6, size_filter=30, n_threads=1)
    return wsfu(ws_input, mask)


def mws(affs, mask):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    strides = [1, 10, 10]
    wsfu = cseg.MutexWatershed(offsets, strides, randomize_bounds=False)
    return wsfu(affs, mask)


# TODO determine threshold
def preagglomerate(affs, ws, n_labels, thresh=.95):
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    probs = nrag.accumulateAffinityStandartFeatures(rag, affs,
                                                    offsets, numberOfThreads=1)[:, 0]
    agglomerator = cseg.MalaClustering(thresh)
    g = nifty.graph.undirectedGraph(rag.numberOfNodes)
    g.insertEdges(rag.uvIds())
    node_labels = agglomerator(g, probs)
    return nrag.projectScalarNodeDataToPixels(rag, node_labels), int(node_labels.max())


def wsdt_preagglomerated(affs, mask, thresh=.95):
    ws, max_id = wsdt(affs, mask)
    ws, max_id = preagglomerate(affs, ws, max_id + 1, thresh)
    return ws, max_id


def fragmenter_factory(algo):
    if algo == 'wslr':
        return algo, long_range_ws
    elif algo == 'wsdt':
        return algo, wsdt
    elif algo == 'mws':
        return algo, mws
    elif algo == 'wsdt_pre':
        return algo, wsdt_preagglomerated
    else:
        raise AttributeError("No!")


#
# segmentation with multicut based segmenters
#


def segment_mc(ws, affs, n_labels, offsets, return_merged_nodes=False):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    probs = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets, numberOfThreads=1)[:, 0]
    uv_ids = rag.uvIds()

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = 5 * costs.min()

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)

    # relabel the node labels consecutively and make sure that zero
    # is still mapped to zero

    if return_merged_nodes:
        return get_merged_nodes(uv_ids, node_labels)
    else:
        return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def segment_mcrf(ws, affs, n_labels, offsets, rf, return_merged_nodes=False):
    rag = nrag.gridRag(ws, numberOfLabels=int(n_labels), numberOfThreads=1)
    feats = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets, numberOfThreads=1)
    probs = rf.predict_proba(feats)[:, 1]
    uv_ids = rag.uvIds()

    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    costs = mc.probabilities_to_costs(probs)
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = 5 * costs.min()

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)
    if return_merged_nodes:
        return get_merged_nodes(uv_ids, node_labels)
    else:
        return nrag.projectScalarNodeDataToPixels(rag, node_labels)


# FIXME something doesn't lift the gil ?!
def segment_lmc(ws, affs, n_labels, offsets, return_merged_nodes=False):
    rag = nrag.gridRag(ws, numberOfLabels=n_labels, numberOfThreads=1)
    lifted_uvs, local_features, lifted_features = nrag.computeFeaturesAndNhFromAffinities(rag,
                                                                                          affs,
                                                                                          offsets,
                                                                                          numberOfThreads=1)
    uv_ids = rag.uvIds()

    lmc = cseg.LiftedMulticut('kernighan-lin', weight_edges=False)
    local_costs = lmc.probabilities_to_costs(local_features[:, 0])
    local_ignore = (uv_ids == 0).any(axis=1)
    local_costs[local_ignore] = 5 * local_costs.min()

    # we might not have lifted edges -> just solve multicut
    if len(lifted_uvs) == 1 and (lifted_uvs[0] == -1).any():
        mc = cseg.Multicut('kernighan-lin', weight_edges=False)
        graph = nifty.graph.undirectedGraph(n_labels)
        graph.insertEdges(uv_ids)
        node_labels = mc(graph, local_costs)

    else:
        lifted_costs = lmc.probabilities_to_costs(lifted_features[:, 0])
        lifted_ignore = (lifted_uvs == 0).any(axis=1)
        lifted_costs[lifted_ignore] = 5 * lifted_costs.min()
        node_labels = lmc(uv_ids, lifted_uvs, local_costs, lifted_costs)

    if return_merged_nodes:
        return get_merged_nodes(uv_ids, node_labels)
    else:
        return nrag.projectScalarNodeDataToPixels(rag, node_labels)


# TODO also try with single rf learned from both features
def segment_lmcrf(ws, affs, n_labels, offsets, rf_local, rf_lifted, return_merged_nodes=False):
    rag = nrag.gridRag(ws, numberOfLabels=n_labels, numberOfThreads=1)
    lifted_uvs, local_features, lifted_features = nrag.computeFeaturesAndNhFromAffinities(rag,
                                                                                          affs,
                                                                                          offsets,
                                                                                          numberOfThreads=1)
    lmc = cseg.LiftedMulticut('kernighan-lin', weight_edges=False)
    uv_ids = rag.uvIds()

    local_costs = lmc.probabilities_to_costs(rf_local.predict_proba(local_features)[:, 1])
    local_ignore = (uv_ids == 0).any(axis=1)
    local_costs[local_ignore] = 5 * local_costs.min()

    # we might not have lifted edges -> just solve multicut
    if len(lifted_uvs) == 1 and (lifted_uvs[0] == -1).any():
        mc = cseg.Multicut('kernighan-lin', weight_edges=False)
        graph = nifty.graph.undirectedGraph(n_labels)
        graph.insertEdges(uv_ids)
        node_labels = mc(graph, local_costs)

    else:
        lifted_costs = lmc.probabilities_to_costs(rf_lifted.predict_proba(lifted_features)[:, 1])
        lifted_ignore = (lifted_uvs == 0).any(axis=1)
        lifted_costs[lifted_ignore] = 5 * lifted_costs.min()
        node_labels = lmc(uv_ids, lifted_uvs, local_costs, lifted_costs)

    if return_merged_nodes:
        return get_merged_nodes(uv_ids, node_labels)
    else:
        return nrag.projectScalarNodeDataToPixels(rag, node_labels)


def get_merged_nodes(uv_ids, node_labels):
    merge_edges = node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]]
    return uv_ids[merge_edges]


def segmenter_factory(algo, feats, return_merged_nodes=False):
    rf_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/random_forests'
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]

    if algo == 'mc':

        if feats == 'local':
            # affinity based mc
            key = 'mc_affs_not_stitched2'
            segmenter = partial(segment_mc, offsets=offsets, return_merged_nodes=return_merged_nodes)

        elif feats == 'rf':
            # rf based mc
            with open(os.path.join(rf_folder, 'rf_ABC_local_affinity_feats.pkl'), 'rb') as fr:
                rf = pickle.load(fr)
            rf.n_jobs = 1
            key = 'mc_rf_not_stitched'
            segmenter = partial(segment_mcrf, offsets=offsets, rf=rf, return_merged_nodes=return_merged_nodes)

        else:
            raise AttributeError("No!")

    elif algo == 'lmc':

        if feats == 'local':
            # affinity based lmc
            key = 'lmc_affs_not_stitched'
            segmenter = partial(segment_lmc, offsets=offsets, return_merged_nodes=return_merged_nodes)

        elif feats == 'rf':
            # rf based lmc
            with open(os.path.join(rf_folder, 'rf_ABC_local_affinity_feats.pkl'), 'rb') as fr:
                rf1 = pickle.load(fr)
            rf1.n_jobs = 1
            with open(os.path.join(rf_folder, 'rf_ABC_lifted_affinity_feats.pkl'), 'rb') as fr:
                rf2 = pickle.load(fr)
            rf2.n_jobs = 1

            key = 'lmc_rf_not_stitched'
            segmenter = partial(segment_lmcrf, offsets=offsets,
                                rf_local=rf1, rf_lifted=rf2, return_merged_nodes=return_merged_nodes)

        else:
            raise AttributeError("No!")

    else:
        raise AttributeError("No!")
    return key, segmenter
