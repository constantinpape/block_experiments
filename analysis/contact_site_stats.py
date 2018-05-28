import json
import numpy as np
import vigra
import z5py
import nifty.graph.rag as nrag


def get_contact_sites(seg, n_labels, labels_of_interest, n_threads):

    # compute the region adjacency graph and filter for the edges
    # that connect neurites of interest
    rag = nrag.gridRag(seg, numberOfLabels=n_labels,
                       numberOfThreads=n_threads)
    uv_ids = rag.uvIds()

    # find the edges that connect neurites of interest
    edge_mask = np.in1d(uv_ids, labels_of_interest).reshape(uv_ids.shape).all(axis=1)
    pairs = uv_ids[edge_mask]
    n_pairs = len(pairs)
    edge_labels = np.zeros_like(edge_mask, dtype='uint32')
    edge_labels[edge_mask] = np.arange(1, n_pairs+1).astype('uint32')

    # build the edge volume with the edges of interest
    edge_builder = nrag.ragCoordinates(rag, numberOfThreads=n_threads)
    edge_vol = edge_builder.edgesToVolume(edge_labels)

    # run connected components and for each connected component
    # get the association to the connected neurites
    edge_vol_cc = vigra.analysis.labelVolumeWithBackground(edge_vol)
    return edge_vol, edge_vol_cc, pairs


def contact_site_stats(edge_vol, edge_vol_cc, pairs):
    contact_sites, sizes = np.unique(edge_vol_cc, return_counts=True)
    contact_sites = contact_sites[1:]
    sizes = sizes[1:]
    stats = {}
    for ii, site_id in enumerate(contact_sites):
        where_site = np.where(edge_vol_cc == site_id)
        edge_id = edge_vol[where_site][0, 0, 0]
        pair = pairs[edge_id]
        # TODO map to world coordinates
        coord = [np.mean(wheres) for wheres in where_site]
        stats[site_id] = {'id-pair': pair,
                          'size': sizes[ii],
                          'coordinate': coord}
    return stats


def synapse_stats(edge_vol_cc, syn, seg, dist, stats):
    # get synapse ids that have overlap with at least one site:
    site_mask = edge_vol_cc != 0
    syn_ids = np.unique(syn[site_mask])
    syn_stats = {}
    for syn_id in syn_ids:
        syn_mask = syn == syn_id
        where_syn = np.where(syn_mask)

        # measuring the distance for the synapse would
        # only make sense if we intersec
        # syn_dist = dist[where_syn]
        # TODO map to world coordinates
        coord = [np.mean(wheres) for wheres in where_syn]
        syn_stats[syn_id] = {'coordinate': coord}

        # get the contact sites overlapping with this synapse
        site_ids = np.unique(edge_vol_cc[where_syn])
        if site_ids[0] == 0:
            site_ids = site_ids[1:]

        for site_id in site_ids:
            syn_site_stat = {}
            ids = stats[site_id]['id-pair']
            for seg_id in ids:
                seg_mask = seg == seg_id
                overlap = np.logical_and(syn_mask, seg_mask)
                syn_dist = dist[overlap]
                syn_site_stat[seg_id] = {'size': np.sum(overlap),
                                         'mean-distance': np.mean(syn_dist),
                                         'std-distance': np.std(syn_dist)}
            if 'synapses' in stats[site_id]:
                stats[site_id]['synapses'].update({syn_id: syn_site_stat})
            else:
                stats[site_id]['synapses'] = {syn_id: syn_site_stat}
    return stats, syn_stats


def compute_all_contact_site_stats(path, seg_key, labels_of_interest, n_threads,
                                   distance_key,
                                   synapse_key=None):
    f = z5py.File(path)
    ds_seg = f[seg_key]
    ds_seg.n_threads = n_threads

    seg = ds_seg[:]
    n_labels = int(seg.max()) + 1
    edge_vol, edge_vol_cc, pairs = get_contact_sites(seg, n_labels,
                                                     labels_of_interest, n_threads)

    ds_dist = f[distance_key]
    ds_dist.n_threads = n_threads
    stats = contact_site_stats(edge_vol, edge_vol_cc, pairs)

    if synapse_key is not None:
        assert distance_key is not None
        ds_syn = f[synapse_key]
        ds_syn.n_threads = n_threads

        syn = ds_syn[:]
        dist = ds_dist[:]

        stats, syn_stats = synapse_stats(edge_vol_cc, syn, dist, stats)

        # TODO make save paths parameter
        with open('./syn_stats.json', 'w') as f:
            json.dump(syn_stats, f)

    with open('./site_stats.json', 'w') as f:
        json.dump(stats, f)
