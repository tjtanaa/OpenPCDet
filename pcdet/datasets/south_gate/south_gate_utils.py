from __future__ import division
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from numba import cuda, jit
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import time
from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List

import yaml
# from .point_in_polygon_cuda import pip_wna_number
# print(os.getcwd())
import sys
from point_viz.converter import PointvizConverter
# sys.path.append('/home/tan/tjtanaa2/OpenPCDet')
# sys.path.append('/home/tan/tjtanaa2/OpenPCDet/pcdet/datasets/south_gate')
BASE_PATH = '/home/tan/tjtanaa2/OpenPCDet'
# print(sys.path)

config_fileptr = open(BASE_PATH + '/pcdet/datasets/south_gate/config.yaml')
config = yaml.load(config_fileptr, Loader=yaml.FullLoader)

# print("========= Server Config ==========")
# print(config)
# exit()


def rotation_matrix(angle: List[float]) -> Any:
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

def transform(data, T):
    transformed = np.transpose(np.dot(T, np.transpose(data)))
    return transformed

# @cuda.jit
# def wna_number_cuda_jit(points, polygon_vertices, wn):

#     row, col = cuda.grid(2)
#     if row < points.shape[0] and col < polygon_vertices.shape[0]-1:
#         if(polygon_vertices[col, 1] <= points[row,1]):
#             if(polygon_vertices[col+1, 1] > points[row,1]):
#                 if( 
#                     ( 
#                         (polygon_vertices[col+1,0] - polygon_vertices[col,0] ) 
#                         * (points[row,1]  - polygon_vertices[col,1])
#                     - (points[row,0]  -  polygon_vertices[col,0] ) 
#                         * (polygon_vertices[col+1,1] - polygon_vertices[col,1]) 
#                     ) > 0
#                 ):
#                     cuda.atomic.add(wn, row, 1)
        
#         else:
#             if(polygon_vertices[col+1, 1] <= points[row,1]):
#                 if( 
#                     ( 
#                         (polygon_vertices[col+1,0] - polygon_vertices[col,0] ) 
#                         * (points[row,1]  - polygon_vertices[col,1])
#                     - (points[row,0]  -  polygon_vertices[col,0] ) 
#                         * (polygon_vertices[col+1,1] - polygon_vertices[col,1]) 
#                     ) < 0
#                 ):
#                     cuda.atomic.add(wn, row, -1)
#     else:
#         return

# def pip_wna_number(points, polygon_vertices):
#     # Copy the arrays to the device
#     points_global_mem = cuda.to_device(points)
#     polygon_vertices_global_mem = cuda.to_device(polygon_vertices)

#     # print("points.shape: ", points.shape)
#     # Allocate memory on the device for the result
#     wn_global_mem = cuda.device_array((points.shape[0]),dtype=np.int32)


#     # Configure the blocks
#     threadsperblock = (128, 8)
#     blockspergrid_x = int(math.ceil(points.shape[0] / threadsperblock[0]))
#     blockspergrid_y = int(math.ceil(polygon_vertices.shape[0] / threadsperblock[1]))
#     blockspergrid = (blockspergrid_x, blockspergrid_y)

#     # Start the kernel 
#     wna_number_cuda_jit[blockspergrid, threadsperblock](points_global_mem,\
#                         polygon_vertices_global_mem, wn_global_mem)
#     cuda.synchronize()
#     # Copy the result back to the host
#     wn = wn_global_mem.copy_to_host()
#     return_wn = np.copy(wn)
#     cuda.current_context().deallocations.clear()
#     return return_wn

# def pip_wna_mask(points, polygon_vertices, inside=False):
#     points = np.ascontiguousarray(np.copy(points), dtype=np.float32)
#     polygon_vertices = np.ascontiguousarray(np.copy(polygon_vertices), dtype=np.float32)
#     # print(points.flags)
#     # print(points.shape)
#     # print("pip_wna_mask: \n\t", points[:5])
#     # print("pip_wna_mask: \n\t", polygon_vertices[:5])
#     # print("polygon_vertices: ", polygon_vertices.shape)
#     wn = pip_wna_number(points, polygon_vertices)
#     # print("wn: ", np.sum(wn))
#     if inside:
#         return ~ (wn == 0)
#     else:
#         return wn == 0

def pip_wna_mask(points, polygon_vertices, inside=False):

    

    points = np.ascontiguousarray(np.copy(points), dtype=np.float32)
    polygon_vertices = np.ascontiguousarray(np.copy(polygon_vertices), dtype=np.float32)
    # print(points.flags)
    # print(points.shape)
    # print("pip_wna_mask: \n\t", points[:5])
    # print("pip_wna_mask: \n\t", polygon_vertices[:5])
    # print("polygon_vertices: ", polygon_vertices.shape)
    wn = pip_wna_number(points, polygon_vertices)
    # print("wn: ", np.sum(wn))
    if inside:
        return ~ (wn == 0)
    else:
        return wn == 0

def _get_crop_region_mask(point_cloud_np:Any):
    xyz = point_cloud_np[:,:3]
    
    # print(config['PREPROCESSING'].keys())
    if "POINTCLOUD_REGIONS" in config['PREPROCESSING'].keys():
        regions = config['PREPROCESSING']['POINTCLOUD_REGIONS']

        total_invalid_mask = None
        for k, v in regions.items():
            # print(v)
            if "VALID_" in k:
                # crop the region that is NOT interested
                valid_region_mask = (xyz[:,0] > v[0]) & (xyz[:,0] < v[2])
                valid_region_mask &= (xyz[:,1] > v[1]) & (xyz[:,1] < v[3])
                if total_invalid_mask is None:
                    total_invalid_mask = valid_region_mask
                else:
                    total_invalid_mask = total_invalid_mask | valid_region_mask

        return total_invalid_mask
    else:
        return np.ones(xyz.shape[0]).astype(np.bool)
    

def _get_crop_region_pc(point_cloud_np:Any):
    
    if "POINTCLOUD_REGIONS" in config['PREPROCESSING'].keys():
        regions = config['PREPROCESSING']['POINTCLOUD_REGIONS']

        for k, v in regions.items():
            # print(v)
            
            if "INVALID_REGION" in k:
                # print("invalid region: ")
                # print("before point_cloud_np.shape: ", point_cloud_np.shape)
                xyz = point_cloud_np[:,:3]
                # crop the region that is NOT interested
                invalid_region_mask = (xyz[:,0] > v[0]) & (xyz[:,0] < v[2])
                invalid_region_mask &= (xyz[:,1] > v[1]) & (xyz[:,1] < v[3])
                # print("np.sum(invalid_region_mask): ", np.sum(invalid_region_mask))
                point_cloud_np = point_cloud_np[~invalid_region_mask,:]
                # print("after point_cloud_np.shape: ", point_cloud_np.shape)
            elif "VALID_REGION" in k:
                xyz = point_cloud_np[:,:3]
                # crop the region that is NOT interested
                valid_region_mask = (xyz[:,0] > v[0]) & (xyz[:,0] < v[2])
                valid_region_mask &= (xyz[:,1] > v[1]) & (xyz[:,1] < v[3])
                point_cloud_np = point_cloud_np[valid_region_mask,:]


        return point_cloud_np
    else:
        return point_cloud_np

def _get_polygon_mask(point_cloud_xyz_np: Any, polygon_vertices_np: Any, inside: bool):
    return pip_wna_mask(point_cloud_xyz_np, polygon_vertices_np, inside)

def _get_polygon_vertices_from_file(polygon_filepath:str, dims:int):
    polygon_vertices = np.genfromtxt(polygon_filepath, delimiter=',')
    polygon_vertices = polygon_vertices.reshape((-1,3))
    valid_polygon_vertices = np.copy(polygon_vertices[:,:2])
    # return_polygon_vertices = \
    #     np.concatenate([valid_polygon_vertices, valid_polygon_vertices[np.newaxis,0,:]], axis=0)
    # print("_get_polygon_vertices_from_file: \n\t", valid_polygon_vertices.shape)
    return valid_polygon_vertices

def _get_polygon_3D_mask(point_cloud_xyz_np: Any, \
                polygon_vertices_np: Any, \
                height_range_np:Any,
                inside:bool):
                
    # print("_get_polygon_mask: \n\t", point_cloud_xyz_np[:5])
    polygon_mask = _get_polygon_mask(point_cloud_xyz_np, polygon_vertices_np, True)
    # print("polygon_mask: ", np.sum(polygon_mask))
    # print("height_range_np: ", height_range_np)
    height_mask = (point_cloud_xyz_np[:,2] > height_range_np[0]) & \
                    (point_cloud_xyz_np[:,2] < height_range_np[1])
    if inside:
        return polygon_mask & height_mask
    else:
        return ~(polygon_mask & height_mask)


def _get_crop_polygon_3D_pc(point_cloud_np:Any):

    if "POINTCLOUD_REGIONS" in config['PREPROCESSING'].keys():
        regions = config['PREPROCESSING']['POINTCLOUD_REGIONS']
        # print("before pc shape: ", point_cloud_np.shape)

        for k, v in regions.items():
            # print(v)
            if "POLYGON" in k:
                xyz = np.zeros((point_cloud_np.shape[0], 3))
                xyz = point_cloud_np[:,:3]
                
                polygon_filepath = config['PREPROCESSING']['POINTCLOUD_REGIONS'][k]['FILENAME']
                dims = 3
                polygon_vertices = _get_polygon_vertices_from_file(BASE_PATH + "/" + polygon_filepath, dims)
                height_range_np = np.array(config['PREPROCESSING']['POINTCLOUD_REGIONS'][k]['HEIGHTRANGE'])

                
                polygon_vertices[:,0] += config['PREPROCESSING']['POINTCLOUD']['XYZ_OFFSET'][0] 
                polygon_vertices[:,1] += config['PREPROCESSING']['POINTCLOUD']['XYZ_OFFSET'][1] 
                # height_range_np += config['PREPROCESSING']['POINTCLOUD']['XYZ_OFFSET'][2]

                if "INVALID_" in k:
                    region_mask = _get_polygon_3D_mask(xyz, polygon_vertices, height_range_np, False)
                    # print("InvalidMask")
                elif "VALID" in k:
                    # print("ValidMask")
                    region_mask = _get_polygon_3D_mask(xyz, polygon_vertices, height_range_np, True)
                else:
                    raise ValueError
                
                # print("RegionMask: ", np.sum(region_mask))
                # crop the region that is NOT interested
                point_cloud_np = point_cloud_np[region_mask,:]
                # print("after pc shape: ", point_cloud_np.shape)

        return point_cloud_np
    else:
        return point_cloud_np

def _load_single_point_cloud_with_background(filepath: str):
    if(os.path.exists(filepath)):
        # print("filepath: ", filepath)
        point_cloud_np = np.fromfile(filepath, '<f4')


        # print(point_cloud_np.shape)
        # print(len(point_cloud_np.tobytes()))
        # from datetime import datetime
        # start=datetime.now()
        point_cloud_np = np.reshape(point_cloud_np, (-1, config['DATA_INFO']['CHANNELS']))

        # filter the height
        # return filtered_point_cloud_np
        # np.savetxt("transformed_merged_sample.csv", point_cloud_np, delimiter=",")
        xyz = point_cloud_np[:,:3]
        features = point_cloud_np[:,3:]

        T_rotate = rotation_matrix(config['PREPROCESSING']['POINTCLOUD']['RXYZ_OFFSET'] )
        xyz = transform(xyz, T_rotate)
        xyz[:,0] += config['PREPROCESSING']['POINTCLOUD']['XYZ_OFFSET'][0]
        xyz[:,1] += config['PREPROCESSING']['POINTCLOUD']['XYZ_OFFSET'][1]
        xyz[:,2] += config['PREPROCESSING']['POINTCLOUD']['XYZ_OFFSET'][2]

        # print("====== diff in xyz ===== ",
        # "\nmean: ", np.mean(np.abs(hard_coded_xyz - xyz)),
        # "\min: ", np.min(np.abs(hard_coded_xyz - xyz)), 
        # "\nmax: ", np.max(np.abs(hard_coded_xyz - xyz)))

        point_cloud_np = np.concatenate([ xyz.astype('<f4'), features.astype('<f4')], axis=1).astype('<f4')

        return point_cloud_np
    else:
        raise FileNotFoundError

def _preprocess_point_cloud(point_cloud_np, bg_remove, crop_area):
    # filtered_point_cloud_np = point_cloud_np
    # filtered_point_cloud_np = _get_crop_region_pc(point_cloud_np)

    # if bg_remove:
    #     invalid_point_mask = _get_remove_background_using_statistics_mask(point_cloud_np)
    #     # print("invalid_point_mask: ", np.sum(invalid_point_mask))
    #     point_cloud_np = point_cloud_np[~invalid_point_mask,:]
    # else:
    #     print("NO BG remove")

    if crop_area:
        point_cloud_np = _get_crop_region_pc(point_cloud_np)
        # print("pointcloudnp: ", point_cloud_np.shape)
        point_cloud_np = _get_crop_polygon_3D_pc(point_cloud_np)
    else:
        print("NO CROP")

    # if "DOWNSAMPLING" in config['PREPROCESSING'].keys():
    #     point_cloud_np = downsample_point_cloud(point_cloud_np, \
    #                         strategy_id = config['PREPROCESSING']['DOWNSAMPLING']['STRATEGY'], \
    #                         params=config['PREPROCESSING']['DOWNSAMPLING']['PARAMS'])
        
    # # filtered_point_cloud_np = _get_valid_region(filtered_point_cloud_np)
    # if "DENOISE" in config['PREPROCESSING'].keys():
    #     filter_indexes = _get_denoising_point_cloud_index(point_cloud_np,
    #                         nb_neighbors=config['PREPROCESSING']["DENOISE"]['PARAMS']["NBR_NEIGHBORS"], 
    #                         std_ratio=config['PREPROCESSING']["DENOISE"]['PARAMS']["STD"])
    #     point_cloud_np = point_cloud_np[filter_indexes,:]

    # # # get the mask of those that are located at (x,y,z) = (0,0,z) (in bird eye view)
    # # mask = np.all(np.abs(filtered_point_cloud_np[:,:2]) < 0.1, axis=1)
    # # filtered_point_cloud_np = filtered_point_cloud_np[~mask,:]
    # # print("filtered_point_cloud_np.shape: ", filtered_point_cloud_np.shape)

    # point_cloud_np = np.reshape(point_cloud_np, -1)
    return point_cloud_np

def _retrieve_binary_data_file(filepath, bg_remove, crop_area):

    point_cloud_np = _load_single_point_cloud_with_background(filepath)
    # filtered_point_cloud_np = point_cloud_np
    # filtered_point_cloud_np = _get_crop_region_pc(point_cloud_np)

    # if bg_remove:
    #     invalid_point_mask = _get_remove_background_using_statistics_mask(point_cloud_np)
    #     # print("invalid_point_mask: ", np.sum(invalid_point_mask))
    #     point_cloud_np = point_cloud_np[~invalid_point_mask,:]
    # else:
    #     print("NO BG remove")

    if crop_area:
        point_cloud_np = _get_crop_region_pc(point_cloud_np)
        # print("pointcloudnp: ", point_cloud_np.shape)
        point_cloud_np = _get_crop_polygon_3D_pc(point_cloud_np)
    else:
        print("NO CROP")

    # if "DOWNSAMPLING" in config['PREPROCESSING'].keys():
    #     point_cloud_np = downsample_point_cloud(point_cloud_np, \
    #                         strategy_id = config['PREPROCESSING']['DOWNSAMPLING']['STRATEGY'], \
    #                         params=config['PREPROCESSING']['DOWNSAMPLING']['PARAMS'])
        
    # # filtered_point_cloud_np = _get_valid_region(filtered_point_cloud_np)
    # if "DENOISE" in config['PREPROCESSING'].keys():
    #     filter_indexes = _get_denoising_point_cloud_index(point_cloud_np,
    #                         nb_neighbors=config['PREPROCESSING']["DENOISE"]['PARAMS']["NBR_NEIGHBORS"], 
    #                         std_ratio=config['PREPROCESSING']["DENOISE"]['PARAMS']["STD"])
    #     point_cloud_np = point_cloud_np[filter_indexes,:]

    # # # get the mask of those that are located at (x,y,z) = (0,0,z) (in bird eye view)
    # # mask = np.all(np.abs(filtered_point_cloud_np[:,:2]) < 0.1, axis=1)
    # # filtered_point_cloud_np = filtered_point_cloud_np[~mask,:]
    # # print("filtered_point_cloud_np.shape: ", filtered_point_cloud_np.shape)

    # point_cloud_np = np.reshape(point_cloud_np, -1)
    # response = make_response(point_cloud_np.tobytes())
    # response.headers.set('Content-Type', 'application/octet-stream')
    # response.headers.set('Content-Disposition', 'attachment', filename='np-array.bin')
    # output_path = "/home/tan/tjtanaa2/OpenPCDet/visualization/southgate_preprocessing_crop"
    # Converter = PointvizConverter(home=output_path)
    # Converter.compile(task_name=filepath.split("/")[-1],
    #                 coors=point_cloud_np[:,[0,1,2]],
    #                 intensity=point_cloud_np[:,3])
    return point_cloud_np

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
    'mediumtruck':         6,
    'bigtruck':            6,
    'cylindrical-truck':   7,
    'crane-truck':         8,
    'motorbike':           9,
    'dontcare':            10,
    'pedestrian':          10,
}

det_obj_list = ["Private Cars",
                "Double Decker Bus",
                "Single Decker Bus",
                "Taxi",
                "Mini Bus",
                "Vans",
                "Heavy Trucks",
                "Cylindrical Trucks",
                "Crane Truck",
                "Motorbikes",
                "Misc"]

id_labeller_category = {}
for k, v in category_dict.items():
    id_labeller_category[v] = k


def convert_bbox_attrs_to_labeller_json(bbox_attrs):
    """convert bbox attrs:
        [w, l, h, x, y, z, r, c, conf, cls_conf] (M, 10) to

        {
            "bounding_boxes": [
                {
                    "center": {
                        "x": -2.498486420893925,
                        "y": -3.6137045337629434,
                        "z": 3.1512131094932556
                    },
                    "width": 11.240801926859085,
                    "length": 2.4579750460653993,
                    "height": 3.5392638444900513,
                    "angle": -0.023629636978551565,
                    "object_id": "coachbus"
                }
            ]
        }
    Args:
        bbox_attrs ([type]): [description]
    """

    output_json={"bounding_boxes":[]}

    for bid in range(len(bbox_attrs)):
        w, l, h, x, y, z, r, c, output_conf, output_cls_conf = bbox_attrs[bid]
        bbox_dict = {}

        bbox_dict["center"] = {
            "x": x,
            "y": y,
            "z": z
        }
        bbox_dict["width"] = w
        bbox_dict["length"] = l
        bbox_dict["height"] = h
        bbox_dict["angle"] = r
        bbox_dict["object_id"] = id_labeller_category[c]
        output_json["bounding_boxes"].append(bbox_dict)
    
    return output_json