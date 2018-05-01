import os
import stat
import fileinput
from shutil import copy, rmtree


# https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
def replace_shebang(file_path, shebang):
    for i, line in enumerate(fileinput.input(file_path, inplace=True)):
        if i == 0:
            print(shebang, end='')
        else:
            print(line, end='')


def make_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def make_batch_jobs_step0(path, out_key, cache_folder, n_jobs, block_shape,
                          executable, eta=1, script_file='jobs_step0.sh'):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'scripts/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    with open(script_file, 'w') as f:

        command = './0_prepare.py %s %s %s %i --block_shape %s' % \
                  (path, out_key, cache_folder, n_jobs,
                   ' '.join(map(str, block_shape)))
        log_file = 'logs/log_blockwise_segmentation_step0.log'
        err_file = 'error_logs/err_blockwise_segmentation_step0.err'
        f.write('bsub -J blockwise_segmentation_step0 -We %i -o %s -e %s \'%s\' \n' %
                (eta, log_file, err_file, command))

    make_executable(script_file)


def make_batch_jobs_step1(path, out_key, cache_folder, n_jobs, block_shape, halo,
                          executable, eta, rf_path='', script_file='jobs_step1.sh'):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'scripts/1_blockwise_segmentation.py'), cwd)
    replace_shebang('1_blockwise_segmentation.py', shebang)
    make_executable('1_blockwise_segmentation.py')

    with open(script_file, 'w') as f:

        for job_id in range(n_jobs):
            command = './1_blockwise_segmentation.py %s %s %s %i --block_shape %s --halo %s' % \
                      (path, out_key, cache_folder, job_id,
                       ' '.join(map(str, block_shape)),
                       ' '.join(map(str, halo)))
            if rf_path != '':
                command += '--rf_path %s' % rf_path
            log_file = 'logs/log_blockwise_segmentation_step1_%i.log' % job_id
            err_file = 'error_logs/err_blockwise_segmentation_step1_%i.err' % job_id
            f.write('bsub -J blockwise_segmentation_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                    (job_id, eta, log_file, err_file, command))

    make_executable(script_file)


def make_batch_jobs_step2(cache_folder, n_jobs,
                          executable, eta, script_file='jobs_step2.sh'):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'scripts/2_compute_block_offsets.py'), cwd)
    replace_shebang('2_compute_block_offsets.py', shebang)
    make_executable('2_compute_block_offsets.py')

    with open(script_file, 'w') as f:

        command = './2_compute_block_offsets.py %s %i' % \
                  (cache_folder, n_jobs)
        log_file = 'logs/log_blockwise_segmentation_step2.log'
        err_file = 'error_logs/err_blockwise_segmentation_step2.err'
        f.write('bsub -J blockwise_segmentation_step2 -We %i -o %s -e %s \'%s\' \n' %
                (eta, log_file, err_file, command))

    make_executable(script_file)


def make_batch_jobs_step3(path, out_key, cache_folder, n_jobs, block_shape, halo,
                          executable, eta, rf_path='', script_file='jobs_step3.sh'):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'scripts/3_blockwise_stitching.py'), cwd)
    replace_shebang('3_blockwise_stitching.py', shebang)
    make_executable('3_blockwise_stitching.py')

    with open(script_file, 'w') as f:

        for job_id in range(n_jobs):
            command = './3_blockwise_stitching.py %s %s %s %i --block_shape %s --halo %s' % \
                      (path, out_key, cache_folder, job_id,
                       ' '.join(map(str, block_shape)),
                       ' '.join(map(str, halo)))
            if rf_path != '':
                command += '--rf_path %s' % rf_path
            log_file = 'logs/log_blockwise_segmentation_step3_%i.log' % job_id
            err_file = 'error_logs/err_blockwise_segmentation_step3_%i.err' % job_id
            f.write('bsub -J blockwise_segmentation_step3_%i -We %i -o %s -e %s \'%s\' \n' %
                    (job_id, eta, log_file, err_file, command))

    make_executable(script_file)


def make_batch_jobs_step4(cache_folder, n_jobs,
                          executable, eta, script_file='jobs_step4.sh'):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'scripts/4_compute_node_assignment.py'), cwd)
    replace_shebang('4_compute_node_assignment.py', shebang)
    make_executable('4_compute_node_assignment.py')

    with open(script_file, 'w') as f:

        command = './4_compute_node_assignment.py %s %i' % \
                  (cache_folder, n_jobs)
        log_file = 'logs/log_blockwise_segmentation_step4.log'
        err_file = 'error_logs/err_blockwise_segmentation_step4.err'
        f.write('bsub -J blockwise_segmentation_step4 -We %i -o %s -e %s \'%s\' \n' %
                (eta, log_file, err_file, command))

    make_executable(script_file)


def make_batch_jobs_step5(path, out_key, cache_folder, n_jobs, block_shape,
                          executable, eta, script_file='jobs_step5.sh'):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'scripts/5_write_stitched_segmentation.py'), cwd)
    replace_shebang('5_write_stitched_segmentation.py', shebang)
    make_executable('5_write_stitched_segmentation.py')

    with open(script_file, 'w') as f:

        for job_id in range(n_jobs):
            command = './5_write_stitched_segmentation.py %s %s %s %i --block_shape %s' % \
                      (path, out_key, cache_folder, job_id,
                       ' '.join(map(str, block_shape)))
            log_file = 'logs/log_blockwise_segmentation_step5_%i.log' % job_id
            err_file = 'error_logs/err_blockwise_segmentation_step5_%i.err' % job_id
            f.write('bsub -J blockwise_segmentation_step5_%i -We %i -o %s -e %s \'%s\' \n' %
                    (job_id, eta, log_file, err_file, command))

    make_executable(script_file)


def make_batch_jobs(path, out_key, cache_folder, n_jobs,
                    block_shape, halo, executable,
                    rf_path='', eta=10):

    assert isinstance(eta, (int, list, tuple))
    if isinstance(eta, (list, tuple)):
        assert len(eta) == 5
        assert all(isinstance(ee, int) for ee in eta)
        eta_ = eta
    else:
        eta_ = (eta,) * 5

    # clean logs
    if os.path.exists('error_logs'):
        rmtree('error_logs')
    os.mkdir('error_logs')

    if os.path.exists('logs'):
        rmtree('logs')
    os.mkdir('logs')

    make_batch_jobs_step0(path, out_key, cache_folder,
                          n_jobs, block_shape)

    make_batch_jobs_step1(path, out_key, cache_folder,
                          n_jobs, block_shape, halo,
                          executable, eta_[0], rf_path=rf_path)

    make_batch_jobs_step2(cache_folder, n_jobs,
                          executable, eta=eta_[1])

    make_batch_jobs_step3(path, out_key, cache_folder,
                          n_jobs, block_shape, halo,
                          executable, eta_[2], rf_path=rf_path)

    make_batch_jobs_step4(cache_folder, n_jobs,
                          executable, eta=eta_[3])

    make_batch_jobs_step5(path, out_key, cache_folder,
                          n_jobs, block_shape,
                          executable, eta_[4],)
