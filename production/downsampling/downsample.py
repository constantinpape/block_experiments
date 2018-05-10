from cremi_tools.sampling import create_label_multiset, downsample_labels


def ds_labels(block_id, key='mc_glia_rf_affs_global'):
    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/raw' % block_id
    in_key = 'segmentations/results/' + key
    out_key = 'segmentations/multires/' + key + '_multires'
    block_size = [256, 256, 26]
    print("Creating label multiset")
    create_label_multiset(path, in_key, out_key, block_size)

    print("Downsampling labels")
    m_best = [4]
    sampling_factors = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2]]
    block_sizes = [[256, 256, 26], [128, 128, 26], [128, 128, 26], [128, 128, 13]]
    downsample_labels(path, out_key, sampling_factors, block_sizes,
                      m_best)


def ds_cremi(sample):
    path = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    in_key = 'segmentations/mc_glia_rf_affs_global'
    out_key = 'segmentations/mc_glia_rf_affs_global_multiscale'
    block_size = [256, 256, 26]
    print("Creating label multiset")
    create_label_multiset(path, in_key, out_key, block_size)

    print("Downsampling labels")
    m_best = [4]
    sampling_factors = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2]]
    block_sizes = [[256, 256, 26], [128, 128, 26], [128, 128, 26], [128, 128, 13]]
    downsample_labels(path, out_key, sampling_factors, block_sizes,
                      m_best)


if __name__ == '__main__':
    # blocks = (1, 2, 3, 4)
    blocks = (2,)
    # for key in ('mc_glia_rf_affs_global',):  # , 'mc_glia_global'):
    for key in ('mc_glia_global',):
        for block_id in blocks:
            ds_labels(block_id, key)

    # for sample in ('C+',):
    #     ds_cremi(sample)
