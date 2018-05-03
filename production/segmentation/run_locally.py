import os
from concurrent import futures
from subprocess import call


def step_a(script, path, out_key, cache_folder, job_id, block_shape, halo, rf_path):
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


def step_b(script, path, out_key, cache_folder, job_id, block_shape):
    command = ['python', script,
               path, out_key, cache_folder, str(job_id)]
    command.append('--block_shape')
    command.extend(list(map(str, block_shape)))
    call(command, shell=False)


def run_pipeline(path, out_key, cache_folder,
                 n_jobs, max_workers,
                 block_shape, halo, rf_path):

    print("Step 0: prepare volumes and jobs")
    # Zeroth step:
    # Prepare out volumes and block to job assignment
    prepare_command = ['python', 'scripts/0_prepare.py', path, out_key, cache_folder, str(n_jobs)]
    prepare_command.append('--block_shape')
    prepare_command.extend(list(map(str, block_shape)))
    call(prepare_command)

    print("Step 1: segment individual blocks")
    # First step:
    # Compute the blockwise segmentations
    with futures.ProcessPoolExecutor(max_workers) as pp:
        tasks = [pp.submit(step_a, 'scripts/1_blockwise_segmentation.py',
                           path, out_key + '_blocked',
                           cache_folder, job_id,
                           block_shape, halo, rf_path)
                 for job_id in range(n_jobs)]
        [t.result() for t in tasks]

    print("Step 2: compute and add block offsets")
    # Second step:
    # Compute block offsets
    call(['python', 'scripts/2_compute_block_offsets.py', cache_folder, str(n_jobs)])

    # Second A step:
    # Write the offsets to segmentation
    # (Ideally we don't need to do this)
    with futures.ProcessPoolExecutor(max_workers) as pp:
        tasks = [pp.submit(step_b, 'scripts/2a_write_offsets.py',
                           path, out_key + '_blocked',
                           cache_folder, job_id, block_shape)
                 for job_id in range(n_jobs)]
        [t.result() for t in tasks]

    print("Step 3: find node assignments to stitch blocks")
    # Third step:
    # Find node assignments to stitch blocks by
    # running multicut on the node overlaps
    with futures.ProcessPoolExecutor(max_workers) as pp:
        tasks = [pp.submit(step_a, 'scripts/3_blockwise_stitching.py',
                           path, out_key + '_blocked',
                           cache_folder, job_id,
                           # block_shape, halo, rf_path)
                           block_shape, halo, '')
                 for job_id in range(n_jobs)]
        [t.result() for t in tasks]

    print("Step 4: merge node assignments for final segmentation")
    # Fourth step:
    # Compute the final segmentation according to the overlap solutions
    call(['python', 'scripts/4_compute_node_assignment.py', cache_folder, str(n_jobs)])

    print("Step 5: write final segmentation")
    # Fifth step:
    # Write the stitched segmentation
    with futures.ProcessPoolExecutor(max_workers) as pp:
        tasks = [pp.submit(step_b, 'scripts/5_write_stitched_segmentation.py',
                           path, out_key,
                           cache_folder, job_id, block_shape)
                 for job_id in range(n_jobs)]
        [t.result() for t in tasks]


def segment_lauritzen(block_id, use_rf):

    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/raw' % block_id
    cache_folder = '/nrs/saalfeld/papec/cache/cache_lauritzen_%i' % block_id

    max_workers = 60
    n_jobs = 60

    block_shape = (52, 512, 512)
    halo = (5, 50, 50)

    rf_path = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419/rf.pkl'if use_rf else ''
    out_key = 'segmentations/mc_glia_affs'
    if use_rf:
        out_key += '_rf'

    run_pipeline(path, out_key, cache_folder, n_jobs, max_workers,
                 block_shape, halo, rf_path)


def segment_test(use_rf):

    path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/lauritzen/production_test_block.n5'
    cache_folder = '/nrs/saalfeld/papec/cache/cache_lauritzen_0'

    max_workers = 8
    n_jobs = 8

    block_shape = (52, 512, 512)
    halo = (5, 50, 50)

    rf_path = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419/rf.pkl'if use_rf else ''
    out_key = 'segmentations/mc_glia_affs'
    if use_rf:
        out_key += '_rf'

    run_pipeline(path, out_key, cache_folder, n_jobs, max_workers,
                 block_shape, halo, rf_path)


if __name__ == '__main__':
    # segment_lauritzen(1, False)
    # quit()
    for block_id in (2, 3, 4):
        segment_lauritzen(block_id, False)
    for block_id in (1, 2, 3, 4):
        segment_lauritzen(block_id, True)

    # segment_test(True)
