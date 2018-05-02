#! /usr/bin/python

import os
import time
import argparse
import subprocess


def wait_for_jobs(user, max_wait_time=None):
    t_start = time.time()
    while True:
        time.sleep(5)
        n_running = subprocess.check_output(['bjobs | grep %s | wc -l' % user], shell=True).decode()
        n_running = int(n_running.strip('\n'))
        if n_running == 0:
            break
        if max_wait_time is not None:
            t_wait = time.time() - t_start
            if t_wait > max_wait_time:
                print("MAX WAIT TIME EXCEEDED")
                break


def wait_and_check_multiple_jobs(job_prefix, n_jobs, user='papec'):

    success_marker = 'Success'
    wait_for_jobs(user)

    jobs_failed = []
    for job_id in range(n_jobs):
        log_file = './logs/log_%s_%i.log' % (job_prefix, job_id)

        have_log = os.path.exists(log_file)
        if not have_log:
            jobs_failed.append(job_id)
            continue

        with open(log_file, 'r') as f:
            out = f.readline()
            have_success = out[:len(success_marker)] == success_marker
        if not have_success:
            jobs_failed.append(job_id)
            continue

    return jobs_failed


def wait_and_check_single_job(job_name, user='papec'):

    success_marker = 'Success'
    wait_for_jobs(user)

    log_file = './logs/log_%s.log' % job_name

    job_failed = False
    have_log = os.path.exists(log_file)
    if not have_log:
        job_failed = True

    with open(log_file, 'r') as f:
        out = f.readline()
        have_success = out[:len(success_marker)] == success_marker
    if not have_success:
        job_failed = True

    return job_failed


# TODO retrial for failed jobs
def run_cluster_jobs(n_jobs):

    t_tot = time.time()

    # submit jobs 0
    subprocess.call(['./jobs_step0.sh'])
    failed_jobs = wait_and_check_single_job('blockwise_segmentation_step0')
    if failed_jobs:
        print("Step 0 failed")
        return

    # submit jobs 1
    subprocess.call(['./jobs_step1.sh'])
    failed_jobs = wait_and_check_multiple_jobs('blockwise_segmentation_step1', n_jobs)
    if failed_jobs:
        print("Step 1 failed for following jobs:")
        print(failed_jobs)
        return

    # submit jobs 2
    subprocess.call(['./jobs_step2.sh'])
    failed_jobs = wait_and_check_single_job('blockwise_segmentation_step2')
    if failed_jobs:
        print("Step 2 failed")
        return

    # submit jobs 3
    subprocess.call(['./jobs_step3.sh'])
    failed_jobs = wait_and_check_multiple_jobs('blockwise_segmentation_step3', n_jobs)
    if failed_jobs:
        print("Step 3 failed for following jobs:")
        print(failed_jobs)
        return

    # submit jobs 4
    subprocess.call(['./jobs_step4.sh'])
    failed_jobs = wait_and_check_single_job('blockwise_segmentation_step4')
    if failed_jobs:
        print("Step 4 failed")
        return

    # submit jobs 5
    subprocess.call(['./jobs_step5.sh'])
    failed_jobs = wait_and_check_multiple_jobs('blockwise_segmentation_step5', n_jobs)
    if failed_jobs:
        print("Step 5 failed for following jobs:")
        print(failed_jobs)
        return

    t_tot = time.time() - t_tot
    print("All jobs finished successfully in %f s" % t_tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    run_cluster_jobs(args.n_jobs)
