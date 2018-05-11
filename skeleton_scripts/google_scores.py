import os
import json
from cremi_tools.skeletons import build_skeleton_metrics


# TODO add run-length computation
def compute_google_scores(block_id, seg_key, skeleton_postfix):
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    label_file = os.path.join(path, 'raw', 'segmentations', 'results', seg_key)
    skel_group = 'neurons_of_interest' if skeleton_postfix == '' else 'for_eval_%s' % skeleton_postfix

    skeleton_file = os.path.join(path, 'skeletons', skel_group)

    n_threads = 40
    metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)

    correct, split, merge, n_merges = metrics.computeGoogleScore(n_threads)
    print("Overall edge scores:")
    print("Correct:     ", correct)
    print("Split:       ", split)
    print("Merge:       ", merge)
    print("Merge Points:", n_merges)

    return {'correct': correct, 'split': split, 'merge': merge, 'n_merges': n_merges}


def full_evaluation(block_id, segmentation_keys, out_file):
    scores = {}
    postfixes = ['', '20180508']
    for seg_key in segmentation_keys:
        for postfix in postfixes:
            print("Running eval for", seg_key, postfix)
            score = compute_google_scores(block_id,
                                          seg_key,
                                          postfix)
            scores[seg_key + '_' + postfix] = score
        print()
    with open(out_file, 'w') as f:
        json.dump(scores, f, indent=4, sort_keys=True)


# TODO debug
def test():
    block_id = 2
    seg_key = 'mc_glia_rf_affs_global'
    # postfix = ''
    postfix = '20180508'
    compute_google_scores(block_id, seg_key, postfix)

    # path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    # label_file = os.path.join(path, 'raw', 'segmentations', 'results', seg_key)
    # skel_group = 'neurons_of_interest' if postfix == '' else 'for_eval_%s' % postfix

    # skeleton_file = os.path.join(path, 'skeletons', skel_group)

    # n_threads = 40
    # metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)

    # split_score = metrics.computeSplitScores(1)
    # for skel_id, val in split_score.items():
    #     print("Skeleton", skel_id, ":", val)


if __name__ == '__main__':
    # test()

    seg_keys = ['mc_glia_affs', 'mc_glia_affs_rf',
                'mc_glia_global', 'mc_glia_rf_affs_global']
    full_evaluation(2, seg_keys, 'eval_block2.json')
