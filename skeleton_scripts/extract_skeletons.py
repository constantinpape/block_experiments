import os
import sys
import numpy as np
# sys.path.append('/home/papec/Work/my_projects/cremi_tools')
# sys.path.append('/home/papec/Work/my_projects/z5/bld/python')
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/cremi_tools')


def intersect_skeletons_with_bb(skeletons, bb):
    # intersect the skeletons with our bounding box
    intersecting_skeletons = {}
    bb_offset = np.array([bb[i].start for i in range(3)])
    for skel_id, values in skeletons.items():
        coords = values['coordinates']
        in_bb = np.concatenate([np.logical_and(coords[:, i] >= bb[i].start,
                                               coords[:, i] < bb[i].stop)[:, None] for i in range(3)],
                               axis=1)
        in_bb = np.all(in_bb, axis=1)
        if in_bb.any():
            intersecting_coords = coords[in_bb] - bb_offset
            # find the intersecting edges by checking which of the potential edges in the bounding
            # box are in range
            nodes = values['node_ids'][in_bb]
            edges = values['edges'][in_bb]

            # TODO properly check for invalid edges
            # valid_edges = np.in1d(edges, nodes).reshape(edges.shape)
            # invalid_edges = np.logical_not(valid_edges.all(axis=1))
            # edges[invalid_edges] = np.array([-1, -1], dtype='int64')[:, None]

            intersecting_skeletons[skel_id] = {'coordinates': intersecting_coords, 'node_ids': nodes, 'edges': edges}

    print("Found", len(intersecting_skeletons), "intersecting skeletons")
    return intersecting_skeletons


def extract_skeletons(block_id, skeleton_postfix='', with_names=False):
    from cremi_tools.skeletons import SkeletonParserCSV
    # path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/skeletons/skeletons-0%i.csv' % (block_id, block_id)
    if skeleton_postfix == '':
        path = '/nrs/saalfeld/lauritzen/0%i/skeletons.csv' % block_id
    else:
        path = '/nrs/saalfeld/lauritzen/0%i/skeletons-%s.csv' % (block_id, skeleton_postfix)
    assert os.path.exists(path), path
    # TODO correct resolution and offsets ?!
    resolution = (4, 4, 40)
    offsets = (0, 0, 0)

    # TODO this part should go into cremi_tools.skeletons
    parser = SkeletonParserCSV(resolution=resolution,
                               offsets=offsets,
                               invert_coordinates=True,
                               have_name_column=with_names)
    skeleton_dict = parser.parse(path)
    skel_ids = np.array(skeleton_dict['skeleton_ids'], dtype='uint64')
    node_ids = np.array(skeleton_dict['node_ids'], dtype='uint64')
    parents = np.array(skeleton_dict['parents'], dtype='int64')
    coords = np.array(skeleton_dict['coordinates'], dtype='int64')
    if with_names:
        names = skeleton_dict['names']
    assert (coords >= 0).all()

    # seperate by individual neurons
    extracted_skeletons = {}
    unique_ids = np.unique(skel_ids)
    for skid in unique_ids:
        extracted = {}
        sk_mask = skel_ids == skid

        extracted['coordinates'] = coords[sk_mask]
        if with_names:
            extracted['name'] = names[skid]

        nodes = node_ids[sk_mask]
        parent_nodes = parents[sk_mask]
        edges = np.concatenate([nodes[:, None].astype('int64'),
                                parent_nodes[:, None]], axis=1)
        extracted['node_ids'] = nodes
        extracted['edges'] = edges

        extracted_skeletons[skid] = extracted
    return extracted_skeletons


# serialize the skeletons properly to n5
def save_skeletons(block_id, skeletons, skeleton_postfix=''):
    import z5py
    # path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/skeletons' % block_id
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5/skeletons' % block_id

    f = z5py.File(path)
    # if we don't have the post-fix, these are
    # the initial neurons of interest
    if skeleton_postfix == '':
        fg = f.create_group('neurons_of_interest')
    # otherwise, thse are for evaluation
    else:
        fg = f.create_group('for_eval_%s' % skeleton_postfix)

    for skel_id, values in skeletons.items():
        g = fg.create_group(str(skel_id))
        # we prepend the nodes to the coordinates
        nodes = values['node_ids']
        coords = values['coordinates']
        coords = np.concatenate([nodes[:, None], coords], axis=1)
        dsc = g.create_dataset('coordinates', shape=coords.shape,
                               chunks=coords.shape, dtype='uint64',
                               compression='raw')
        dsc[:] = coords.astype('uint64')
        # save the edges
        edges = values['edges']
        dse = g.create_dataset('edges', shape=edges.shape,
                               chunks=edges.shape, dtype='int64',
                               compression='raw')
        dse[:] = edges.astype('int64')
        # save the name as attribute if present
        if 'name' in values:
            g.attrs['name'] = values['name']


def view_skeletons(block_id, skeleton_postfix=''):
    import z5py
    from cremi_tools.viewer.volumina import view
    from cremi_tools.skeletons import visualize_skeletons

    # skeletons = extract_skeletons(block_id)
    skel_path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/skeletons' % block_id
    assert os.path.exists(skel_path), skel_path

    f_skel = z5py.File(skel_path)
    # if we don't have the post-fix, these are
    # the initial neurons of interest
    if skeleton_postfix == '':
        fg = f_skel['neurons_of_interest']
    # otherwise, thse are for evaluation
    else:
        fg = f_skel['for_eval_%s' % skeleton_postfix]

    print("Loading skeletons...")
    skeletons = {}
    for skel_id in fg.keys():
        if not skel_id.isdigit():
            continue
        g = fg[skel_id]
        coords = g['coordinates'][:]
        node_ids = coords[:, 0]
        coords = coords[:, 1:]
        edges = g['edges'][:]
        skeletons[int(skel_id)] = {'coordinates': coords.astype('uint64'),
                                   'node_ids': node_ids,
                                   'edges': edges}
    print("... done")

    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5' % block_id
    f = z5py.File(path)
    ds = f['filtered/gray']

    shape = ds.shape
    central = tuple(sh // 2 for sh in shape)
    offset = (100, 1000, 1000)
    bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))

    print("Visualizing skeletons ...")
    bb_shape = tuple(b.stop - b.start for b in bb)
    skeletons = intersect_skeletons_with_bb(skeletons, bb)
    skeleton_vol = visualize_skeletons(bb_shape, skeletons)
    print("... done")

    print("Have skeletons, loading raw from bb", bb)
    raw = ds[bb]
    view([raw, skeleton_vol])


def get_gt_roi(gt_id):
    resolution = (40., 4., 4.)
    padded_offset = ((12000., 2608., 9172.),
                     (0., 8076., 7056.),
                     (8000., 4308., 4708.))
    offset_in_padded = (1480.0, 3644.0, 3644.0)

    offset = tuple(int((pad_off + off) // res)
                   for pad_off, off, res in zip(padded_offset[gt_id],
                                                offset_in_padded,
                                                resolution))

    shape = (125, 1250, 1250)
    return offset, tuple(off + sh
                         for off, sh in zip(offset, shape))


if __name__ == '__main__':
    from cremi_tools.skeletons import filter_skeletons_in_rois

    block_id = 2
    skeleton_postfix = '20180508'
    with_name = False
    print("Extracting skeletons from csv...")
    skeletons = extract_skeletons(block_id, skeleton_postfix, with_name)
    print("... done")

    # we filter the skeletons for the regions where we have training
    # data in block 2
    if block_id == 2:
        print("Filtering skeleton locations ...")
        # regions of the new groudntruh block rois for block 2
        roi_list = [get_gt_roi(gt_id) for gt_id in range(3)]
        skeletons = filter_skeletons_in_rois(skeletons, roi_list)
        print("... done")

    print("Saving to n5 ...")
    save_skeletons(block_id, skeletons, skeleton_postfix)
    print("... done")

    # skeleton_postfix = '20180508'
    # view_skeletons(2, skeleton_postfix)
