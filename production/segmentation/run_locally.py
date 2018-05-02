import os
from concurrent import futures
from subprocess import call


def step(script, path, out_key, cache_folder, job_id, block_shape, halo, rf_path):
    assert os.path.exists(script)
    assert script[-3:] == '.py'
    command = ['python', script,
               path, out_key, cache_folder, str(job_id)]
    command.append('--block_shape')
    command.extend(list(map(str, block_shape)))
    command.append('--halo')
    command.extend(list(map(str, halo)))
    if rf_path != '':
        command.append('--rf_path')
        command.append(rf_path)
    call(command, shell=False)


def step5(path, out_key, cache_folder, job_id, block_shape):
    command = ['python', 'scripts/5_write_stitched_segmentation.py',
               path, out_key, cache_folder, str(job_id)]
    command.append('--block_shape')
    command.extend(list(map(str, block_shape)))
    call(command, shell=False)


def run_pipeline(path, out_key, cache_folder,
                 n_jobs, max_workers,
                 block_shape, halo, rf_path):

    # Zeroth step:
    # Prepare out volumes and block to job assignment
    prepare_command = ['python', 'scripts/0_prepare.py', path, out_key, cache_folder, str(n_jobs)]
    prepare_command.append('--block_shape')
    prepare_command.extend(list(map(str, block_shape)))
    call(prepare_command)

    # First step:
    # Compute the blockwise segmentations
    with futures.ProcessPoolExecutor(max_workers) as pp:
        tasks = [pp.submit(step, 'scripts/1_blockwise_segmentation.py',
                           path, out_key,
                           cache_folder, job_id,
                           block_shape, halo, rf_path)
                 for job_id in range(n_jobs)]
        [t.result() for t in tasks]

    # Second step:
    # Compute block offsets
    call(['python', 'scripts/2_compute_block_offsets.py', cache_folder, str(n_jobs)])

    # Third step:
    # Find node assignments to stitch blocks by
    # running multicut on the node overlaps
    with futures.ProcessPoolExecutor(max_workers) as pp:
        tasks = [pp.submit(step, 'scripts/3_blockwise_stitching.py',
                           path, out_key,
                           cache_folder, job_id,
                           block_shape, halo, rf_path)
                 for job_id in range(n_jobs)]
        [t.result() for t in tasks]

    # Fourth step:
    # Compute the final segmentation according to the overlap solutions
    call(['python', 'scripts/4_compute_node_assignment.py', cache_folder, str(n_jobs)])

    # Fifth step:
    # Write the stitched segmentation
    with futures.ProcessPoolExecutor(max_workers) as pp:
        tasks = [pp.submit(step5, path, out_key,
                           cache_folder, job_id, block_shape)
                 for job_id in range(n_jobs)]
        [t.result() for t in tasks]


if __name__ == '__main__':
    block_id = 1
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/raw' % block_id

    # path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/lauritzen/production_test_block.n5'

    cache_folder = '/nrs/saalfeld/papec/cache/cache_lauritzen_%i' % block_id

    max_workers = 40
    n_jobs = 40

    block_shape = (52, 512, 512)
    halo = (5, 50, 50)

    rf_path = ''
    out_key = 'segmentations/mc_glia_affs'
    run_pipeline(path, out_key, cache_folder, n_jobs, max_workers,
                 block_shape, halo, rf_path)
