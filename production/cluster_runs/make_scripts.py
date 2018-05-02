import sys
sys.path.append('../segmentation')


def make_scripts(block_id, n_jobs, block_shape, halo):
    from make_cluster_scripts import make_batch_jobs
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/raw' % block_id
    cache_folder = '/nrs/saalfeld/papec/cache/cache_lauritzen_%i' % block_id
    eta = [20, 5, 10, 5, 10]
    make_cluster_scripts(path, 'segmentations/mc_glia_affs', cache_folder, n_jobs,
                         block_shape, halo, executable, eta)


if __name__ == '__main__':
    block_id = 2
    n_jobs = 400
    block_shape = (52, 512, 512)
    halo = (5, 50, 50)
    make_scripts(block_id, n_jobs, block_shape, halo)
