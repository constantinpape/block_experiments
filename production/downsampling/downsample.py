from cremi_tools.sampling import create_label_multiset, downsample_labels


def ds_labels(block_id, key='segmentations/mc_glia_affs'):
    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/raw' % block_id
    out_key = key + '_multires'
    block_size = [256, 256, 26]
    print("Creating label multiset")
    create_label_multiset(path, key, out_key, block_size)

    print("Downsampling labels")
    m_best = [5]
    sampling_factors = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2]]
    block_sizes = [[256, 256, 26], [128, 128, 26], [128, 128, 26], [128, 128, 13]]
    downsample_labels(path, out_key, sampling_factors, block_sizes,
                      m_best)


if __name__ == '__main__':
    # ds_labels(1)
    for block_id in (2, 3, 4):
        ds_labels(block_id)
