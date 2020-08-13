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
import json
import pickle

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



class DemoStreamingDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def prepare_data(self, point_cloud):
        input_dict = {
            'points': point_cloud,
            'frame_id': 0,
        }

        print("prepare data")
        return super().prepare_data(data_dict = input_dict)



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_path', type=str, default='./run/', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    save_path = os.path.join(args.save_path, datetime.now().strftime("%m-%d-%Y=%H-%M-%S"))
    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger
    # )
    demo_streaming_dataset = DemoStreamingDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_streaming_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    # print("done loading")
    model.cuda()
    # print("done adding model to gpu")
    # model.cpu()
    model.eval()
    # print("done converting model to inference mode")

    context = zmq.Context()

    # Socket to talk to server
    print("Connecting to hello world server...")
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:5555")

    time_taken = []
    num_test_points = 110
    with torch.no_grad():
        idx = 0
        WIDTH = 2048
        HEIGHT = 128
        N_POINTS = WIDTH * HEIGHT
        while idx < num_test_points:
            print("idx: ", idx)
            # point_cloud = np.random.normal(3, 2.5, size=(N_POINTS, 4))
            # point_cloud[:,3] = 0
            # [0, -39.68, -3, 69.12, 39.68, 1]
            # _x = np.random.uniform(0, 69.12, size=(N_POINTS, 1))
            # _y = np.random.uniform(-39.68,39.68 , size=(N_POINTS, 1))
            # _z = np.random.uniform(-3, 1, size=(N_POINTS, 1))
            # _i = np.zeros((N_POINTS, 1))
            # point_cloud = np.concatenate([_x, _y,_z,_i], axis=1)
            # print("pointcloud.shape: ", point_cloud.shape)
            start = timer()

            message = socket.recv()
            # print(type(message))
            point_cloud = np.frombuffer(message, dtype=np.float32)
            # print(point_cloud.shape)
            point_cloud = np.reshape(point_cloud,(-1,4))

            data_dict = demo_streaming_dataset.prepare_data(point_cloud)
            
            logger.info(f'Process sample index: \t{idx + 1}')
            # print("data_dict: ", data_dict)
            data_dict = demo_streaming_dataset.collate_batch([data_dict])
            # print("data_dict: ", data_dict)
            # print("data_dict[points].shape: " ,data_dict['points'].shape)
            # print("max data_dict[points][0]: " , np.max(data_dict['points'][:,0]), "min data_dict[points][0]: " , np.min(data_dict['points'][:,0]))
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print("pred_dict: ", pred_dicts)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # mlab.show(stop=True)

            interested_data = {}
            interested_data['points'] = data_dict['points'].cpu().numpy()
            interested_data['pred'] = []
            for i, data in enumerate(pred_dicts):
                interested_data['pred'].append({})
                interested_data['pred'][i]['pred_boxes'] = pred_dicts[i]['pred_boxes'].cpu().numpy()
                interested_data['pred'][i]['pred_scores'] = pred_dicts[i]['pred_scores'].cpu().numpy()
                interested_data['pred'][i]['pred_labels'] = pred_dicts[i]['pred_labels'].cpu().numpy()
            data_dict_path = os.path.join(save_path, "data_{}.pkl".format(idx))
            # print(interested_data)
            # np.save(data_dict_path, interested_data)
            with open(data_dict_path, 'wb') as f:
                pickle.dump(interested_data, f)
            
            end = timer()
            idx += 1
            if idx <= 100: # skip 100 readings which acts as a warm up
                continue
            time_taken.append(end-start)
            
        
        # print("average_time_taken (s): " ,timedelta(seconds=np.mean(time_taken)))
        # print("std_time_taken (s): " ,timedelta(seconds=np.std(time_taken)))
        # print("min time taken (s): ", timedelta(seconds=np.min(time_taken)))
        # print("max time taken (s): ", timedelta(seconds=np.max(time_taken)))

        print("total number of time stamps: ", len(time_taken))
        mean = np.mean(np.array(time_taken))
        print("mean: ", timedelta(seconds=mean))
        std = np.std(np.array(time_taken))
        print("std: ", timedelta(seconds=std))
        min_v = np.min(np.array(time_taken))
        print("min: ", timedelta(seconds=min_v))
        max_v = np.max(np.array(time_taken))
        print("max: ",timedelta(seconds= max_v))
        median_v = np.median(np.array(time_taken))
        print("median: ", timedelta(seconds=median_v))
        # First quartile (Q1) 
        Q1 = np.percentile(np.array(time_taken), 25, interpolation = 'midpoint') 
        print("Q1: ", timedelta(seconds=Q1))
        # Third quartile (Q3) 
        Q3 = np.percentile(np.array(time_taken), 75, interpolation = 'midpoint') 
        print("Q3: ", timedelta(seconds=Q3))
        # Interquaritle range (IQR) 
        IQR = Q3 - Q1 
        print("IQR: ", timedelta(seconds=IQR) )
        
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
