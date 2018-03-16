import os
import sys
import numpy as np
sys.path.append('/home/papec/Work/my_projects/cremi_tools')
sys.path.append('/home/papec/Work/my_projects/z5/bld/python')


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


def extract_skeletons(block_id):
    from cremi_tools.skeletons import SkeletonParserCSV
    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5/skeletons/skeletons-0%i.csv' % (block_id, block_id)
    assert os.path.exists(path)
    # TODO correct resolution and offsets ?!
    resolution = (4, 4, 40)
    offsets = (0, 0, 0)
    parser = SkeletonParserCSV(resolution=resolution,
                               offsets=offsets,
                               invert_coordinates=True)
    skeleton_dict = parser.parse(path)
    skel_ids = np.array(skeleton_dict['skeleton_ids'], dtype='uint64')
    node_ids = np.array(skeleton_dict['node_ids'], dtype='int64')
    parents = np.array(skeleton_dict['parents'], dtype='int64')
    coords = np.array(skeleton_dict['coordinates'], dtype='int64')
    names = skeleton_dict['names']
    assert (coords >= 0).all()

    # seperate by individual neurons
    extracted_skeletons = {}
    unique_ids = np.unique(skel_ids)
    for skid in unique_ids:
        extracted = {}
        sk_mask = skel_ids == skid

        extracted['coordinates'] = coords[sk_mask]
        extracted['name'] = names[skid]

        nodes = node_ids[sk_mask]
        parents = node_ids[sk_mask]
        edges = np.concatenate([nodes[:, None], parents[:, None]], axis=1)
        extracted['node_ids'] = nodes
        extracted['edges'] = edges

        extracted_skeletons[skid] = extracted
    return extracted_skeletons


# TODO serialize the skeletons properly to n5
def save_skeletons(block_id):
    pass


def view_skeletons(block_id):
    import z5py
    from cremi_tools.viewer.volumina import view
    from cremi_tools.skeletons import visualize_skeletons

    skeletons = extract_skeletons(block_id)
    path = '/home/papec/mnt/nrs/lauritzen/0%i/workspace.n5' % block_id
    f = z5py.File(path)
    ds = f['filtered/gray']

    shape = ds.shape
    central = tuple(sh // 2 for sh in shape)
    offset = (100, 1000, 1000)
    bb = tuple(slice(c - off, c + off) for c, off in zip(central, offset))

    bb_shape = tuple(b.stop - b.start for b in bb)
    skeletons = intersect_skeletons_with_bb(skeletons, bb)
    skeleton_vol = visualize_skeletons(bb_shape, skeletons)

    print("Have skeletons, loading raw from bb", bb)
    raw = ds[bb]
    view([raw, skeleton_vol])


if __name__ == '__main__':
    # view_skeletons(2)
    save_skeletons(2)
