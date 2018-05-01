import matplotlib.pyplot as plt
import os
import numpy as np
# import seaborn as sns
import pickle
from scipy.stats import norm
from scipy.stats import normaltest

import z5py
from cremi_tools.skeletons import smooth_sliding_window, smooth_bfs


def test_fit(dinstances, skel_id, node_id):
    data = distances[skel_id][node_id]
    # d, p = kstest(data, 'norm')
    chi, p = normaltest(data)
    return chi, p


def plot_histogram_and_gaussian(skel_id, node_id, data, fm_nodes, show=True, save_folder=None):
    labels = ['not-smoothed', 'sliding-window', 'breadth-first']
    # data = distances[skel_id][node_id]
    n_plots = len(data)
    # fig, ax = plt.subplots(1, n_plots, figsize=(12, 12))
    fig, ax = plt.subplots(n_plots, figsize=(12, 12))
    for ii, dd in enumerate(data):
        d = dd[skel_id][node_id]
        n_pix = len(d)
        mu, sigma = norm.fit(d)
        dmax = np.max(d)
        if dmax < 2000:
            drange = (0, 2000)
        else:
            drange = None
        n, bins, patches = ax[ii].hist(d, bins=64, density=True, range=drange)
        y = norm.pdf(bins, mu, sigma)
        ax[ii].plot(bins, y, linewidth=2)
        if ii == n_plots - 1:
            ax[ii].set_xlabel('Distance [nm]')
        ax[ii].set_title('%s: mean = %.2f, sigma = %.2f, n_pix=%i' % (labels[ii], mu, sigma, n_pix))

    if show:
        plt.show()

    else:
        assert save_folder is not None
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        have_fm = False
        if skel_id in fm_nodes:
            if node_id in fm_nodes[skel_id]:
                have_fm = True

        if have_fm:
            fold = os.path.join(save_folder, 'explicit_merge')
            if not os.path.exists(fold):
                os.mkdir(fold)
            plt.savefig(os.path.join(fold, 'skel%i_node%i.png' % (skel_id, node_id)))

        else:
            fold = os.path.join(save_folder, 'no_explicit_merge')
            if not os.path.exists(fold):
                os.mkdir(fold)
            plt.savefig(os.path.join(fold, 'skel%i_node%i.png' % (skel_id, node_id)))

    plt.close()


def smooth_distance_stats(distances, window=2):
    skeleton_folder = '/home/papec/mnt/nrs/lauritzen/02/workspace.n5/skeletons'
    skeleton_ids = os.listdir(skeleton_folder)
    smoothed_distances = {}
    for skel_id in skeleton_ids:
        if not skel_id.isdigit():
            continue
        skel_path = os.path.join(skeleton_folder, skel_id)
        edges = z5py.File(skel_path)['edges'][:]
        skel_id = int(skel_id)
        smoothed_distances[skel_id] = smooth_sliding_window(distances[skel_id], edges, window)
    return smoothed_distances


def smooth_distance_stats_bfs(distances, window=2):
    skeleton_folder = '/home/papec/mnt/nrs/lauritzen/02/workspace.n5/skeletons'
    skeleton_ids = os.listdir(skeleton_folder)
    smoothed_distances = {}
    for skel_id in skeleton_ids:
        if not skel_id.isdigit():
            continue
        skel_path = os.path.join(skeleton_folder, skel_id)
        edges = z5py.File(skel_path)['edges'][:]
        skel_id = int(skel_id)
        smoothed_distances[skel_id] = smooth_bfs(distances[skel_id], edges, window)
    return smoothed_distances


def plot_smoothed_histograms(distances, fm_nodes):
    for window in (2, 3, 4, 6, 10, 15):
        save_folder = 'plots_w%i' % window
        print("Smoothing distance statistics with sliding window")
        sliding_distances = smooth_distance_stats(distances, window)
        print("Smoothing distance statistics with bfs")
        bfs_distances = smooth_distance_stats_bfs(distances, window)
        print("Generating plots for window size:", window)
        for skel_id in distances:
            for node_id in distances[skel_id]:
                plot_histogram_and_gaussian(skel_id, node_id,
                                            [distances, sliding_distances, bfs_distances],
                                            fm_nodes, False, save_folder)


if __name__ == '__main__':
    with open('skeleton_distance_cache.pkl', 'rb') as f:
        distances = pickle.load(f)
    with open('fm_nodes.pkl', 'rb') as f:
        fm_nodes = pickle.load(f)
    plot_smoothed_histograms(distances, fm_nodes)
