import os
from cremi_tools.skeletons import SkeletonParserSWC, save_skeletons, skeletons_from_swc_to_n5_format


def extract_skeletons(paths, save_path, save_group):
    # pixel resolution in nanometer
    resolution = (4, 4, 40)
    # Sample E offset in nanometer
    offsets = (376000, 80000, 158200)
    parser = SkeletonParserSWC(resolution=resolution,
                               offsets=offsets,
                               invert_coordinates=True)
    print("Extracting skeletons from swc")
    skeletons = skeletons_from_swc_to_n5_format(parser, paths)
    print("Saving skeletons to n5")
    save_skeletons(save_path, save_group, skeletons)


if __name__ == '__main__':
    top_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/sampleE/skeletons'
    sub_folders = ['v14_KC', 'v14_PN']
    paths = []
    for fold in sub_folders:
        paths.extend([os.path.join(top_path, fold, fname)
                      for fname in os.listdir(os.path.join(top_path, fold))])
    save_group = 'n5-skeletons'
    extract_skeletons(paths, top_path, save_group)
