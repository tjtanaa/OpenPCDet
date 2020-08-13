import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
# print("a")
from pcdet.datasets import DatasetTemplate
# print("b")
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
import mayavi.mlab as mlab
from timeit import default_timer as timer
from datetime import timedelta, datetime
import os
import zmq
import pickle



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_path', type=str, default='./run/08-13-2020=18-48-09', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    print("load save_path: ", args.save_path)
    if not os.path.exists(args.save_path):
        print("ERROR file not found")
        exit()

    filenames = []
    if os.path.isdir(args.save_path):
        for files in os.listdir(args.save_path):
            if '.npy' in files or '.pkl' in files or '.json' in files:
                print("add ", os.path.join(args.save_path, files), " to_list")
                filenames.append(os.path.join(args.save_path, files))
                
    else:
        if '.npy' in files or '.pkl' in files or '.json' in files:
            print("add ", files, " to_list")
            filenames.append(args.save_path)

    
    for filename in filenames:
        # data_dict = np.load(filename, allow_pickle=True)
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            # print(dir(data_dict))
            # print(data_dict)
        # print(data_dict.data['points'])
        # print(type(data_dict))
        points = torch.from_numpy(data_dict['points'])
        print(points)
        pred_dicts = []
        for i, bbox in enumerate(data_dict.get('pred')):
            pred_dicts.append({})
            pred_dicts[i]['pred_boxes'] = torch.from_numpy(data_dict.get('pred')[i]['pred_boxes'])
            pred_dicts[i]['pred_scores'] = torch.from_numpy(data_dict.get('pred')[i]['pred_scores'])
            pred_dicts[i]['pred_labels'] = torch.from_numpy(data_dict.get('pred')[i]['pred_labels'])

            # pred_dicts =  torch.from_numpy(data_dict.get('pred'))
        V.draw_scenes(
            points=points[:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )
        mlab.show(stop=True)

    # with torch.no_grad():
    #     for idx, data_dict in enumerate(demo_dataset):
    #         logger.info(f'Visualized sample index: \t{idx + 1}')
    #         print("data_dict: ", data_dict)
    #         data_dict = demo_dataset.collate_batch([data_dict])
    #         print("data_dict: ", data_dict)
    #         # print("data_dict[points].shape: " ,data_dict['points'].shape)
    #         # print("max data_dict[points][0]: " , np.max(data_dict['points'][:,0]), "min data_dict[points][0]: " , np.min(data_dict['points'][:,0]))
    #         load_data_to_gpu(data_dict)
    #         pred_dicts, _ = model.forward(data_dict)
    #         print("pred_dict: ", pred_dicts)
    #         # V.draw_scenes(
    #         #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
    #         #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
    #         # )
    #         # mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
