# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate

from . import south_gate_utils

from numpy.linalg import multi_dot
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List
import json
from point_viz.converter import PointvizConverter

# category_dict = {
#     'two-box':             0,
#     'three-box':           0,
#     'dd':                  1,
#     'coachbus':            2,
#     'taxi':                3,
#     'publicminibus':       4,
#     'privateminibus':      4,
#     'one-box':             5,
#     'one-box-panel-back':  5,
#     'gogovan':             5,
#     'smalltruck':          6,
#     'mediumtruck':         7,
#     'bigtruck':            7,
#     'motorbike':           8,
#     'crane-truck':         9,
#     'cylindrical-truck':   10,
#     'dontcare':            11,
#     'pedestrian':          11
# }

# det_obj_list = ["Private Cars",
#                 "Double Decker Bus",
#                 "Single Decker Bus",
#                 "Taxi",
#                 "Mini Bus",
#                 "Vans",
#                 # "Light Trucks",
#                 "Heavy Trucks",
#                 # "Motorbikes",
#                 "Crane Truck",
#                 "Cylindrical Truck",
#                 "Misc"]
category_dict = {
    'two-box':             0,
    'three-box':           0,
    'dd':                  1,
    'coachbus':            2,
    'taxi':                3,
    'publicminibus':       4,
    'privateminibus':      4,
    'one-box':             5,
    'one-box-panel-back':  5,
    'gogovan':             5,
    'smalltruck':          6,
    'mediumtruck':         6,
    'bigtruck':            6,
    'motorbike':           7,
    'crane-truck':         8,
    'cylindrical-truck':   9,
    # 'dontcare':            11,
    # 'pedestrian':          11
}

det_obj_list = ["Private Cars",
                "Double Decker Bus",
                "Single Decker Bus",
                "Taxi",
                "Mini Bus",
                "Vans",
                "Light Trucks",
                # "Heavy Trucks",
                "Motorbikes",
                "Crane Truck",
                "Cylindrical Truck"]



def get_rotation_matrix(angle):
    """ This is a function that generates the rotation matrix.
    Args:
        angle (List[float]): List of [rx, ry, rz] in radians
    Returns:
        Tuple[Any, Any]: The rotation matrix
    """
    
    T = np.eye(3)
    
    rx = angle[0]
    Rx = np.array([[1, 0 , 0],
                  [0, np.cos(rx),  -np.sin(rx)],
                  [0, np.sin(rx), np.cos(rx)]]) # x axis rotation
    ry = angle[1]
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]]) # y axis rotation kitti camera frame
    rz = angle[2]
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                  [np.sin(rz), np.cos(rz), 0],
                  [0, 0, 1]]) # z axis rotation
    
    T = multi_dot([Rz, Ry, Rx, T])
    return T


def _retrieve_list_of_directories(datapath):

    config={}
    config['DATA_SOURCE'] = datapath

    if os.path.exists(config['DATA_SOURCE']):
        directory_list = [directory for directory in os.listdir(config['DATA_SOURCE'])
                                    if (os.path.isdir(os.path.join(config['DATA_SOURCE'], directory)) and "_dir" in directory)  ]
        # print(directory_list)
        # print(len(directory_list))

        charbefore = 20

        def func(x):
            return x[:charbefore]+x[charbefore:][:-4].zfill(4)
        # print(directory_list)
        # print(len(filename_list))
        directory_list = sorted(directory_list,key=func)

    return directory_list
    # return jsonify(directory_list=directory_list)


def _retrieve_list_of_data_files(datapath, directory):

    config={}
    config['DATA_SOURCE'] = datapath
    if os.path.exists(config['DATA_SOURCE']):
        # print(os.listdir(os.path.join(config['DATA_SOURCE'], directory)))
        # print(len(os.listdir(os.path.join(config['DATA_SOURCE'], directory))))
        filename_list = [filename for filename in os.listdir(os.path.join(config['DATA_SOURCE'], directory))
                                    if (os.path.isfile(
                                            os.path.join(config['DATA_SOURCE'], 
                                            os.path.join(directory, filename)
                                            )) and ".bin" in filename)  ]
        # print(directory_list)
        # print(len(filename_list))
        # import glob
        # filelist = glob.glob(BINARY_DIR+"*.bin")

        charbefore = 20

        def func(x):
            return x[:charbefore]+x[charbefore:][:-4].zfill(4)
        # print(directory_list)
        # print(len(filename_list))
        filename_list = sorted(filename_list,key=func)

    return filename_list
    # return jsonify(filename_list=filename_list)

def get_all_filenames(path):
    directory_list = _retrieve_list_of_directories(path)

    filename_list = []
    for directory in directory_list:
        file_list= [
            os.path.join(directory, filename) for
            filename in _retrieve_list_of_data_files(path, directory)
        ]
        filename_list.extend(
            file_list
        )
    return filename_list

def generate_split_txt(path, split_ratio):

    split_path = os.path.join(path,'ImageSets')

    if not os.path.exists(split_path):
        os.makedirs(split_path)

    split_name_list = None

    if len(split_ratio) == 1:
        split_name_list = ['train', 'val']
    elif len(split_ratio) == 2:
        split_name_list = ['train', 'val']
    elif len(split_ratio) == 3:
        split_name_list = ['train', 'val', 'test']
    
    data_path = os.path.join(path, "Data")

    filename_list = get_all_filenames(data_path)
    num_files = len(filename_list)

    # cum_ratio = np.sum(split_ratio)
    if len(split_ratio) == 1:
        for idx, split_name in enumerate(split_name_list):
            list_path = os.path.join(split_path, split_name + '.txt')
            with open(list_path, 'w') as f:
                # f.write('\n'.join(filename_list))
                
                for relativefilepath in filename_list:
                    json_file = os.path.join(path,'Label', relativefilepath + '.json')
                    with open(json_file) as fjson:
                        json_label = json.load(fjson)
                        valid_obj_count=0
                        for i in range(len(json_label['bounding_boxes'])):
                            bbox_json = json_label['bounding_boxes'][i]
                            if bbox_json['object_id'] not in category_dict.keys():
                                continue
                            #     object_list.append(bbox_json['object_id'])
                            #     print(bbox_json['object_id'])
                            valid_obj_count+=1
                        if valid_obj_count > 0:
                            # f.write('\n'.join(filename_list))
                            f.write(relativefilepath+'\n')


    else:
        print("There are %d samples in total" % (num_files))
        split_indices = [0]
        cum_ratio = 0.0
        for ratio in split_ratio:
            cum_ratio += ratio
            split_indices.append(int(cum_ratio * num_files))

        assert int(cum_ratio) == 1

        for idx, split_name in enumerate(split_name_list):
            list_path = os.path.join(split_path, split_name + '.txt')
            with open(list_path, 'w') as f:
                f.write('\n'.join(filename_list[
                    split_indices[idx]:split_indices[idx+1]
                ]))

def get_union_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
    return output

def trim(data):
    keep_idx = get_union_sets([
            data[:, 0] > -50.32, 
            data[:, 0] < 30.32,
            data[:, 1] > -50.32, 
            data[:, 1] < 30.32,
            data[:, 2] > 0.0,
            data[:, 2] < 8.0])

    return data[keep_idx, :]


tr_m = get_rotation_matrix([0.0705718, -0.2612746, -0.017035])


class SouthGateDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.data_path = self.root_path 
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # print(split_dir)
        # exit()
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        # print(self.sample_sequence_list)
        self.infos = []
        self.include_south_gate_data(self.mode)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_south_gate_data(self.mode)

    def include_south_gate_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        # for k in range(len(self.sample_sequence_list)):
        #     sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
        #     info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
        #     # info_path = self.check_sequence_name_with_all_version(info_path)
        #     if not info_path.exists():
        #         num_skipped_infos += 1
        #         continue
        #     with open(info_path, 'rb') as f:
        #         infos = pickle.load(f)
        #         waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        # if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
        #     sampled_waymo_infos = []
        #     for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
        #         sampled_waymo_infos.append(self.infos[k])
        #     self.infos = sampled_waymo_infos
        #     self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    # @staticmethod
    # def check_sequence_name_with_all_version(sequence_file):
    #     if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
    #         sequence_file = Path(str(sequence_file)[:-9] + '_with_camera_labels.tfrecord')
    #     if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
    #         sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

    #     return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        import concurrent.futures as futures
        from functools import partial
        # from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        def process_single_scene(relativefilepath):

            print('process %s' % (relativefilepath))
            data_path = os.path.join(self.root_path, 'Data')
            label_path = os.path.join(self.root_path, 'Label')
            info = {}
            pc_info = {'num_features': 4, 'filename': relativefilepath}

            info['point_cloud'] = pc_info
            calib_info = {
                'xyz_offset': np.array([0,0,5.7]),
                'rxyz_offset': np.array([0.0705718,-0.2612746,-0.017035])
            }
            calib_info['rotation_matrix'] = tr_m

            info['calib'] = calib_info

            # print('%s sample_idx: %s' % (self.split, sample_idx))
            # info = {}
            # pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            # info['point_cloud'] = pc_info

            # image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            # info['image'] = image_info
            # calib = self.get_calib(sample_idx)

            # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            # R0_4x4[3, 3] = 1.
            # R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            # info['calib'] = calib_info

            if has_label:
                json_file = os.path.join(label_path, relativefilepath + '.json')
                with open(json_file) as f:
                    json_label = json.load(f)
                    bboxes = []
                for i in range(len(json_label['bounding_boxes'])):
                    bbox_json = json_label['bounding_boxes'][i]
                    if bbox_json['object_id'] not in category_dict.keys():
                        continue
                    #     object_list.append(bbox_json['object_id'])
                    #     print(bbox_json['object_id'])

                    bbox = [bbox_json['width'],
                            bbox_json['length'],
                            bbox_json['height'],
                            bbox_json['center']['x'],
                            bbox_json['center']['y'],
                            bbox_json['center']['z'],
                            bbox_json['angle'],
                            category_dict[bbox_json['object_id']],
                            bbox_json['object_id'], # original label
                            det_obj_list[category_dict[bbox_json['object_id']]], # remapped label
                            0]# difficulty level
                    bboxes.append(bbox)

                annotations = {}
                annotations['name'] = np.array([bbox[9] for bbox in bboxes])
                annotations['dimensions'] = np.array([[bbox[0], bbox[1], bbox[2]] for bbox in bboxes])
                annotations['location'] = np.array([[bbox[3], bbox[4], bbox[5]] for bbox in bboxes])  # lhw(camera) format
                annotations['rotation_y'] = np.array([bbox[6] for bbox in bboxes])

                
                num_objects = len([bbox[9] for bbox in bboxes if bbox[9] != 'Misc'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                # bboxes_np = np.asarray(bboxes)
                annotations['gt_boxes_lidar'] = \
                    np.array([
                        [bbox[3], bbox[4], bbox[5],
                        bbox[0], bbox[1], bbox[2],
                        bbox[6]
                        ] for bbox in bboxes])
                # print(annotations['gt_boxes_lidar'])
                info['annos'] = annotations


            return info

        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, self.sample_sequence_list)
        
        return list(infos)

    def get_lidar(self, absfilepath):

        bin_file = absfilepath
        if not os.path.exists(bin_file):
            # logging.warning("Raw data {} does not exist.".format(bin_file))
            # continue
            print("Raw data {} does not exist.".format(bin_file))
        data = np.fromfile(bin_file, dtype=np.float32).reshape([-1, 7])[:, :4]
        # data = trim(data)
        
        data[:, :3] = np.matmul(tr_m, np.transpose(data[:, :3])).transpose()
        data[:, 2] += 5.7
        # data = south_gate_utils._get_crop_polygon_3D_pc(data)

        data = trim(data)
        return data

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        # pc_info = info['point_cloud']
        # sequence_name = pc_info['lidar_sequence']
        # sample_idx = pc_info['sample_idx']
        # points = self.get_lidar(sequence_name, sample_idx)

        data_path = os.path.join(self.root_path, 'Data')
        label_path = os.path.join(self.root_path, 'Label')

        pc_info = info['point_cloud']
        relativefilepath = pc_info['filename']
        points = self.get_lidar(os.path.join(data_path, relativefilepath))

        input_dict = {
            'points': points,
            'frame_id': 0,
        }

        if 'annos' in info:
            annos = info['annos']
            # annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar
                # 'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        # data_dict['metadata'] = info.get('metadata', info['frame_id'])
        # data_dict.pop('num_points_in_gt', None)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=1,
                                    processed_data_tag=None):
        database_save_path = save_path / ('pcdet_gt_database_%s_sampled_%d' % (split, sampled_interval))
        db_info_save_path = save_path / ('pcdet_south_gate_dbinfos_%s_sampled_%d.pkl' % (split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)


        data_path = os.path.join(self.root_path, 'Data')
        label_path = os.path.join(self.root_path, 'Label')

        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            # sequence_name = pc_info['lidar_sequence']
            # sample_idx = pc_info['sample_idx']
            relativefilepath = pc_info['filename']
            points = self.get_lidar(os.path.join(data_path, relativefilepath))

            annos = info['annos']
            names = annos['name']
            # difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            # print("num_obj: ", num_obj)
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            # print("box_idxs_of_pts.shape: ", box_idxs_of_pts.shape)
            # print(np.unique(box_idxs_of_pts))

            
            # gt_points = points[box_idxs_of_pts > -1]
            # Converter = PointvizConverter(home='/home/tan/tjtanaa2/OpenPCDet/visualization/south_gate_dataloader')
            # Converter.compile(task_name="Pc_Generator_valid",
            #                 coors=gt_points[:,[1,2,0]],
            #                 intensity=gt_points[:,3],
            #                 bbox_params=gt_boxes[:,[4,5,3,1,2,0,6]])

            # exit()
            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (relativefilepath.rstrip('.bin').split('/')[1], names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'filename': relativefilepath,
                               'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': 0}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_south_gate_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='south_gate_processed_data',
                       workers=multiprocessing.cpu_count()):
    
    generate_split_txt(ROOT_DIR, [0.8, 0.2])
    # generate_split_txt(ROOT_DIR, [1.0])
    dataset = SouthGateDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('south_gate_infos_%s.pkl' % train_split)
    val_filename = save_path / ('south_gate_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------South Gate info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------South Gate info val file is saved to %s----------------' % val_filename)


    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=det_obj_list
    )
    print('---------------Data preparation Done---------------')


    # with open(train_filename, 'rb') as f:
    #     train_info = pickle.load(f)
        
    # print(train_info)


    # with open(val_filename, 'rb') as f:
    #     val_info = pickle.load(f)
    
    # print(val_info)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_south_gate_infos', help='')
    args = parser.parse_args()

    ROOT_DIR = Path("/media/data3/tjtanaa/south_gate_1031")
    generate_split_txt(ROOT_DIR, [1.0])

    # exit()
    if args.func == 'create_south_gate_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        ROOT_DIR = Path("/media/data3/tjtanaa/south_gate_1031")
        # generate_split_txt(ROOT_DIR, [0.9,0.1])
        create_south_gate_infos(
            dataset_cfg=dataset_cfg,
            class_names=det_obj_list,
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG
        )

    # Database Private Cars: 126
    # Database Mini Bus: 90
    # Database Misc: 239
    # Database Crane Truck: 18
    # Database Vans: 159
    # Database Double Decker Bus: 42
    # Database Cylindrical Truck: 23
    # Database Heavy Trucks: 17
    # Database Taxi: 40
    # Database Single Decker Bus: 5

    # Database Single Decker Bus: 152
    # Database Mini Bus: 138
    # Database Private Cars: 424
    # Database Taxi: 141
    # Database Vans: 324
    # Database Double Decker Bus: 116
    # Database Light Trucks: 99
    # Database Motorbikes: 23
    # Database Crane Truck: 49
    # Database Cylindrical Truck: 23

    # python -m pcdet.datasets.south_gate.south_gate_dataset --func create_south_gate_infos --cfg_file tools/cfgs/dataset_configs/south_gate_dataset.yaml
