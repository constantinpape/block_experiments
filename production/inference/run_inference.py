import os
import sys
import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.postprocessing import clip_float_to_uint8

# TODO make sure to choose correct folder
networks = {'mala': {'prefix': 'unet_mala',
                     'input_shape': (88, 808, 808),
                     'output_shape': (60, 596, 596)}}


def single_gpu_inference(path, network_key, gpu, iteration):
    assert os.path.exists(path), path

    net_top_folder = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/new_experiments'
    net_sub_folder = 'experiments_unet_mala_mask_glia_predict_glia'
    net_folder = os.path.join(net_top_folder, net_sub_folder)
    assert os.path.exists(net_folder), net_folder

    prefix = networks[network_key]['prefix']
    graph_weights = os.path.join(net_folder, '%s_checkpoint_%i' % (prefix, iteration))
    # we don't use inference model for dtu2
    if network_key == 'dtu2':
        graph_inference = os.path.join(net_folder, prefix)
    else:
        graph_inference = os.path.join(net_folder, '%s_inference' % prefix)
    net_io_json = os.path.join(net_folder, 'net_io_names.json')

    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = './offsets/list_gpu_%i.json' % gpu
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = net_io_names["raw"]
    output_key = net_io_names["affs"]

    input_shape = networks[network_key]['input_shape']
    output_shape = networks[network_key]['output_shape']

    prediction = TensorflowPredict(graph_weights,
                                   graph_inference,
                                   input_key=input_key,
                                   output_key=output_key)

    target_key = 'predictions/affs_glia'
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     clip_float_to_uint8,
                     path, path,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     input_key='gray',
                     target_keys=target_key,
                     num_cpus=10,
                     channel_order=[list(range(13))])
    t_predict = time.time() - t_predict

    with open(os.path.join(path, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    path = sys.argv[1]
    network_key = sys.argv[2]
    gpu = int(sys.argv[3])
    iteration = int(sys.argv[4])
    single_gpu_inference(path, network_key, gpu, iteration)
