import torch
import torch.nn as nn
from torch.autograd import Function
from functools import partial

import numpy as np
# from . 
# import dvp_conv_cuda
from ...ops.dvp_conv.dvp_conv_utils import VoxelSamplingFeatureFunction
from ...ops.dvp_conv.dvp_conv_utils import grid_sampling_gpu
from ...ops.dvp_conv.dvp_conv_utils import voxel_sampling_feature_gpu
from ...ops.dvp_conv.dvp_conv_utils import voxel_sampling_idx_binary_gpu
from ...ops.dvp_conv.dvp_conv_utils import voxel_sampling_idx_gpu

# from point_viz.converter import PointvizConverter
# from tqdm import tqdm

# Converter = PointvizConverter("/media/data2/dvpdet_visualization")

# def plot_points_from_voxels_with_color(voxels, center_coors, resolution, self_rgbs=False, kernel_size=3, mask=-1, name='test'):
#     output_coors = []
#     output_rgb = []
#     half_kernel_size = (kernel_size - 1) / 2
#     for i in tqdm(range(len(voxels))):
#         r, g, b = np.random.randint(low=0, high=255, size=3)
#         for n in range(kernel_size ** 3):
#             intensity = voxels[i, n, 0]
#             if intensity != mask:
#                 x = n % kernel_size
#                 z = n // (kernel_size ** 2)
#                 y = (n - z * kernel_size ** 2) // kernel_size
#                 x_coor = (x - half_kernel_size) * resolution + center_coors[i, 0]
#                 y_coor = (y - half_kernel_size) * resolution + center_coors[i, 1]
#                 z_coor = (z - half_kernel_size) * resolution + center_coors[i, 2]
#                 output_coors.append([x_coor, y_coor, z_coor])
#                 if not self_rgbs:
#                     output_rgb.append([r, g, b])
#                 else:
#                     output_rgb.append(voxels[i, n, :])

#     output_coors, output_rgb = np.array(output_coors), np.array(output_rgb)
#     Converter.compile(coors=output_coors[:,[1,2,0]],
#                     default_rgb=output_rgb,
#                     task_name=name)

# def fetch_instance(input_list, num_list, id=0):
#     accu_num_list = np.cumsum(num_list)
#     if id == 0:
#         return input_list[:num_list[0], ...]
#     else:
#         return input_list[accu_num_list[id - 1]:accu_num_list[id], ...]
class DynamicPointConvBasicBlock(nn.Module):
    def __init__(self,
               input_channels,
               output_channels,
               grid_buffer_size,
               output_pooling_size,
               subsample_res,
               kernel_res,
               dimension=None,
               coor_offset=None,
               activation='relu',
               kernel_size=3,
               padding=0.0,
               bn_decay=None,
               use_bias=False,
               concat_feature=False,
               last_layer=False):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.grid_buffer_size = grid_buffer_size
        self.output_pooling_size = output_pooling_size
        self.activation = activation
        self.dimension = dimension
        self.coor_offset = coor_offset
        self.subsample_res = subsample_res
        self.kernel_res = kernel_res
        self.kernel_size = kernel_size
        self.padding = padding
        self.bn_decay = bn_decay if not last_layer else None
        self.last_layer = last_layer
        self.use_bias = use_bias
        self.concat_feature = concat_feature

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        # self.weight = nn.Parameter(torch.empty(kernel_size ** 3 * input_channels, output_channels), requires_grad=True)
        # if self.use_bias:
        #     self.bias = nn.Parameter(torch.empty(output_channels), requires_grad=True)
        # else:
        #     # You should always register all possible parameters, but the
        #     # optional ones can be None if you want.
        #     self.register_parameter('bias', None)
        
        self.feature_transformation_layer = nn.Linear(kernel_size ** 3 * input_channels, output_channels, bias=self.use_bias)

        self.grid_sampling_method = grid_sampling_gpu
        # (input_coors,
        #           input_num_list,
        #           resolution,
        #           dimension=None,
        #           coor_offset=None)
        self.voxel_sampling_feature_method = VoxelSamplingFeatureFunction.apply
        
        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        norm_fn = partial(nn.LayerNorm, eps=1e-3 )
        self.bn = norm_fn(output_channels)

        if self.activation is not None:
            activation_fn_dict = {'relu': torch.nn.ReLU(),
                                  'elu': torch.nn.ELU(),
                                  'leaky_relu': torch.nn.LeakyReLU()}
            self.activation_fn = activation_fn_dict[self.activation]

        self.init_weights()

    def init_weights(self):
        # print("init DynamicPointConvBasicBlock weights")
        
        # nn.init.kaiming_normal_(self.weight)
        # nn.init.xavier_normal_(self.weight)
        # if self.bias is not None:
        #     nn.init.constant_(self.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
                # print("init batchnorm1d")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-4)
                # nn.init.constant_(m.weight, 1.0)
                if self.use_bias:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_coors,
               input_features,
               input_num_list,
               voxel_idx,
               center_idx,
               mem_saving,
               concat_features=None):    
        # voxel_sampling_method = voxel_sampling_binary if mem_saving else voxel_sampling
        voxel_sampling_idx_method = voxel_sampling_idx_binary_gpu if mem_saving else voxel_sampling_idx_gpu
        input_num_list = input_num_list.type(torch.int32)

        if self.subsample_res is not None:
            kernel_center_coors, center_num_list, center_idx = \
                self.grid_sampling_method(input_coors=input_coors,
                                    input_num_list=input_num_list,
                                    resolution=self.subsample_res,
                                    dimension=self.dimension,
                                    coor_offset=self.coor_offset)
            if self.concat_feature:
                if concat_features is not None:
                    concat_features = concat_features[center_idx.type(torch.int64),:]
        else:
            kernel_center_coors = input_coors
            center_num_list = input_num_list
            center_idx = center_idx

        if self.kernel_res is not None:
            voxel_idx, features = voxel_sampling_idx_method(input_coors=input_coors,
                                                            input_features=input_features,
                                                            input_num_list=input_num_list,
                                                            center_coors=kernel_center_coors,
                                                            center_num_list=center_num_list,
                                                            resolution=self.kernel_res,
                                                            dimension=self.dimension,
                                                            coor_offset=self.coor_offset,
                                                            kernel_size=self.kernel_size,
                                                            grid_buffer_size=self.grid_buffer_size,
                                                            output_pooling_size=self.output_pooling_size)
        else:
            voxel_idx = voxel_idx[center_idx]
            features = input_features
        
        # print("voxelidx: ", voxel_idx)
        # print("features.shape: ", features.shape)
        voxel_features = self.voxel_sampling_feature_method(features,
                                                voxel_idx,
                                                self.kernel_size,
                                                self.padding)

        # print("voxel_features.shape: ", voxel_features.shape)
        # print(voxel_features[0,:])
        # print("self.input_channels: ", self.input_channels)
        voxel_features = torch.reshape(voxel_features, shape=[-1, self.kernel_size ** 3 * self.input_channels])
        # output_features = torch.matmul(voxel_features, self.weight)
        output_features = self.feature_transformation_layer(voxel_features)

        # print("Before: output_features.shape: ", output_features.shape)
        output_feature_list = []
        if self.bn_decay is None and self.use_bias:
            # output_features = output_features + self.bias
            pass
        else:
            center_accum_num_list = torch.cumsum(center_num_list, dim=0)
            output_feature_list.append(self.bn(output_features[None,:center_accum_num_list[0],:])[0,:,:])

            for batch_id in range(1, center_num_list.shape[0]):
                output_feature_list.append(
                    self.bn(output_features[None,center_accum_num_list[batch_id-1]:center_accum_num_list[batch_id],:])[0,:,:])
        # print("weight norm: ", torch.norm(self.weight))
        output_features = torch.cat(output_feature_list,axis=0)
        # print("After: output_features.shape: ", output_features.shape)
        if self.activation is not None:
            output_features = self.activation_fn(output_features)

        # print("output_features.shape: ", output_features.shape)
        if self.concat_feature:
            if concat_features is not None:
                concat_features = torch.cat([concat_features, output_features], axis=1)
            return kernel_center_coors, output_features, center_num_list, voxel_idx, center_idx, concat_features
    
        return kernel_center_coors, output_features, center_num_list, voxel_idx, center_idx

class DynamicPointConvBackBone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        '''
        BACKBONE_3D:
            NAME: DynamicPointConvBackBone
            LAYER_PARAMS:
                PADDING: 0.0
                SUBSAMPLE_RES: [0.1, 0.2, 0.4, 0.8, 1.6]
                OUTPUT_CHANNELS: [16, 32, 64, 128, 256]
                KERNEL_RES: [0.2, 0.4, 0.8, 1.6, 3.2]
                KERNEL_SIZE: [3, 3, 3, 3, 3]
                HEADS: [False, False, False, False, True] # new
            MODEL_PARAMS:
                ACTIVATION: 'relu'
                GRID_BUFFER_SIZE: 3
                OUTPUT_POOLING_SIZE: 5
                SAVE_MEM: true
            DIMENSION_PARAMS:
                DIMENSIONS: null # need to modified to point cloud range here if possible
                COOR_OFFSET: null
 
        '''
        
        self.dvp_conv_layers = nn.ModuleList()

        output_channels = self.model_cfg.LAYER_PARAMS.OUTPUT_CHANNELS.copy()
        subsample_res = self.model_cfg.LAYER_PARAMS.SUBSAMPLE_RES.copy()
        kernel_res = self.model_cfg.LAYER_PARAMS.KERNEL_RES.copy()
        heads = self.model_cfg.LAYER_PARAMS.HEADS.copy()
        kernel_size = self.model_cfg.LAYER_PARAMS.KERNEL_SIZE.copy()
        self.mem_saving = self.model_cfg.MODEL_PARAMS.SAVE_MEM
        self.num_keypoints = self.model_cfg.MODEL_PARAMS.NUM_KEYPOINTS
        # print("kernel_size: ", kernel_size)
        # print("self.model_cfg.LAYER_PARAMS.PADDING: ", self.model_cfg.LAYER_PARAMS.PADDING)
        self.kernel_res = kernel_res

        for k in range(self.model_cfg.LAYER_PARAMS.SUBSAMPLE_RES.__len__()):
            # print("K Layer: ", k )
            self.dvp_conv_layers.append(
                DynamicPointConvBasicBlock(input_channels=output_channels[k],
                        output_channels=output_channels[k+1],
                        grid_buffer_size=self.model_cfg.MODEL_PARAMS.GRID_BUFFER_SIZE,
                        output_pooling_size=self.model_cfg.MODEL_PARAMS.OUTPUT_POOLING_SIZE,
                        subsample_res=subsample_res[k],
                        kernel_res=kernel_res[k],
                        dimension=self.model_cfg.DIMENSION_PARAMS.DIMENSIONS,
                        coor_offset=self.model_cfg.DIMENSION_PARAMS.COOR_OFFSET,
                        activation=self.model_cfg.MODEL_PARAMS.ACTIVATION,
                        kernel_size=kernel_size[k],
                        padding=self.model_cfg.LAYER_PARAMS.PADDING,
                        bn_decay=None,
                        last_layer=heads[k])
            )
        self.num_point_features = output_channels[-1]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        num_points_list = batch_dict['num_points']
        coors, features, num_list = points[:,1:4].contiguous(), points[:,4:].contiguous(), num_points_list
        concat_features = features
        voxel_idx, center_idx = None, None
        
        for i in range(len(self.dvp_conv_layers)):
            coors, features, num_list, voxel_idx, center_idx = \
                self.dvp_conv_layers[i](coors,
                                        features,
                                        num_list,
                                        voxel_idx,
                                        center_idx,
                                        mem_saving=self.mem_saving)
            # print(coors.shape, features.shape, num_list.shape, voxel_idx.shape, center_idx.shape)
        
        # add batch idx to point_coors
        num_sampled_points = torch.tensor(self.num_keypoints, 
            dtype=torch.int32, device=coors.device)
        # batch_id_torch_cuda = torch.zeros((coors.shape[0], 1), 
        #     dtype=torch.float32,
        #     device=coors.device)
        
        accum_torch = torch.cumsum(num_list, dim=0).type(torch.int64)
        resampled_coor = torch.zeros((num_sampled_points * batch_size, 4), 
            dtype=torch.float32,
            device=coors.device)
        resampled_features = torch.zeros((num_sampled_points *batch_size, features.shape[1]), 
            dtype=torch.float32,
            device=features.device)
        
        start_idx = 0
        end_idx = torch.min(num_list[0] , num_sampled_points)

        resampled_coor[:end_idx, 1:] = coors[:end_idx, :]
        resampled_features[:end_idx, :] = features[:end_idx, :]

        for batch_id in range(1, batch_size):
            start_idx = batch_id*num_sampled_points
            end_idx = torch.min(num_list[batch_id] , num_sampled_points)
            # end_idx = np.min([accum_torch[batch_id].detach().cpu().numpy() , num_sampled_points])

            resampled_coor[start_idx: start_idx+end_idx, 1:] = \
                coors[accum_torch[batch_id-1]: (accum_torch[batch_id-1]+end_idx), :]

            resampled_features[start_idx: start_idx+end_idx,:] = \
                features[accum_torch[batch_id-1]: (accum_torch[batch_id-1]+end_idx), :]

            resampled_coor[(batch_id-1)*num_sampled_points:batch_id*num_sampled_points, 0] = batch_id
            # resample_points
            # batch_id_torch_cuda[(batch_id-1)*num_sampled_points:batch_id*num_sampled_points, 0] = batch_id
            # output_voxels = fetch_instance(features.detach().cpu().numpy(), num_list.detach().cpu().numpy(), id=batch_id)
            # output_centers = fetch_instance(coors.detach().cpu().numpy(), num_list.detach().cpu().numpy(), id=batch_id)

            # # plot_points_from_voxels_with_color(voxels=output_voxels[:,:,None],
            # #                                    center_coors=output_centers,
            # #                                    resolution=self.kernel_res[-1],
            # #                                    self_rgbs=True,
            # #                                    name='voxel_sampling')

            # intensity = np.sum(output_voxels, axis=1)
            # Converter.compile(coors=output_centers[:,[1,2,0]],
            #                 intensity=intensity,
            #                 task_name='voxel_sampling' + str(batch_id))
        ##### Duplicate points to reach number of sampled points
        # batch_id_torch_cuda = torch.zeros((batch_size * self.num_keypoints, 1), 
        #     dtype=torch.float32,
        #     device=coors.device)
        
        # accum_torch = torch.cumsum(num_list, dim=0).type(torch.int64)
        # resampled_coor_list = []
        # resampled_features_list = []
        
        # start_idx = 0
        # # end_idx = torch.min(num_list[0] , num_sampled_points)

        # random_idx= torch.randint(low=0,high=num_list[0], size=num_sampled_points, device=coors.device)

        # resampled_coor_list.append(coors[random_idx, :])
        # resampled_features_list.append(features[random_idx, :])

        # for batch_id in range(1, batch_size):
        #     start_idx = batch_id*num_sampled_points
        #     # end_idx = torch.min(num_list[batch_id] , num_sampled_points)
        #     random_idx= torch.randint(low=0,high=num_list[0], size=num_sampled_points, device=coors.device) + accum_torch[batch_id-1]
        #     # end_idx = np.min([accum_torch[batch_id].detach().cpu().numpy() , num_sampled_points])

        #     resampled_coor_list.append(coors[random_idx, :])

        #     resampled_features_list.append(features[random_idx, :])

        #     # resample_points
        #     batch_id_torch_cuda[(batch_id-1)*num_sampled_points:batch_id*num_sampled_points, 0] = batch_id

        # resampled_features = torch.cat(resampled_features_list, axis=0)
        # resampled_coor = torch.cat(resampled_coor_list, axis=0)
        #################

        batch_dict['point_features'] = resampled_features
        # batch_dict['point_coords'] = torch.cat((batch_id_torch_cuda, resampled_coor), dim=1)
        batch_dict['point_coords'] = resampled_coor
        # print(torch.min(resampled_coor, axis=0))
        # print(batch_id_torch_cuda)
        # batch_dict['point_features'] = features
        # batch_dict['point_coords'] = torch.cat((batch_id_torch_cuda, coors), dim=1)
        # print("num_list: ", num_list)
        # print("batch_dict['point_features'].shape: ", batch_dict['point_features'].shape)
        # print("batch_dict['point_coords'].shape: ", batch_dict['point_coords'].shape)
        # print("batch_dict['point_coords']: ", batch_dict['point_coords'][0,:])
        return batch_dict


class VariableDynamicPointConvBackBone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        '''
        BACKBONE_3D:
            NAME: DynamicPointConvBackBone
            LAYER_PARAMS:
                PADDING: 0.0
                SUBSAMPLE_RES: [0.1, 0.2, 0.4, 0.8, 1.6]
                OUTPUT_CHANNELS: [16, 32, 64, 128, 256]
                KERNEL_RES: [0.2, 0.4, 0.8, 1.6, 3.2]
                KERNEL_SIZE: [3, 3, 3, 3, 3]
                HEADS: [False, False, False, False, True] # new
            MODEL_PARAMS:
                ACTIVATION: 'relu'
                GRID_BUFFER_SIZE: 3
                OUTPUT_POOLING_SIZE: 5
                SAVE_MEM: true
            DIMENSION_PARAMS:
                DIMENSIONS: null # need to modified to point cloud range here if possible
                COOR_OFFSET: null
 
        '''
        
        self.dvp_conv_layers = nn.ModuleList()

        output_channels = self.model_cfg.LAYER_PARAMS.OUTPUT_CHANNELS.copy()
        subsample_res = self.model_cfg.LAYER_PARAMS.SUBSAMPLE_RES.copy()
        kernel_res = self.model_cfg.LAYER_PARAMS.KERNEL_RES.copy()
        heads = self.model_cfg.LAYER_PARAMS.HEADS.copy()
        kernel_size = self.model_cfg.LAYER_PARAMS.KERNEL_SIZE.copy()
        self.mem_saving = self.model_cfg.MODEL_PARAMS.SAVE_MEM
        self.num_keypoints = self.model_cfg.MODEL_PARAMS.NUM_KEYPOINTS
        # print("kernel_size: ", kernel_size)
        # print("self.model_cfg.LAYER_PARAMS.PADDING: ", self.model_cfg.LAYER_PARAMS.PADDING)
        self.kernel_res = kernel_res

        for k in range(self.model_cfg.LAYER_PARAMS.SUBSAMPLE_RES.__len__()):
            # print("K Layer: ", k )
            self.dvp_conv_layers.append(
                DynamicPointConvBasicBlock(input_channels=output_channels[k],
                        output_channels=output_channels[k+1],
                        grid_buffer_size=self.model_cfg.MODEL_PARAMS.GRID_BUFFER_SIZE,
                        output_pooling_size=self.model_cfg.MODEL_PARAMS.OUTPUT_POOLING_SIZE,
                        subsample_res=subsample_res[k],
                        kernel_res=kernel_res[k],
                        dimension=self.model_cfg.DIMENSION_PARAMS.DIMENSIONS,
                        coor_offset=self.model_cfg.DIMENSION_PARAMS.COOR_OFFSET,
                        activation=self.model_cfg.MODEL_PARAMS.ACTIVATION,
                        kernel_size=kernel_size[k],
                        padding=self.model_cfg.LAYER_PARAMS.PADDING,
                        bn_decay=None,
                        last_layer=heads[k])
            )
        self.num_point_features = output_channels[-1]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        # batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        num_points_list = batch_dict['num_points']
        coors, features, num_list = points[:,1:4].contiguous(), points[:,4:].contiguous(), num_points_list
        # concat_features = features
        voxel_idx, center_idx = None, None
        
        for i in range(len(self.dvp_conv_layers)):
            coors, features, num_list, voxel_idx, center_idx = \
                self.dvp_conv_layers[i](coors,
                                        features,
                                        num_list,
                                        voxel_idx,
                                        center_idx,
                                        mem_saving=self.mem_saving)

        batch_id_torch_cuda = torch.zeros((coors.shape[0], 1), 
            dtype=torch.float32,
            device=coors.device)
        
        accum_torch = torch.cumsum(num_list, dim=0).type(torch.int64)
        start_id = 0
        for batch_id in range(0, accum_torch.shape[0]):
            # resample_points
            end_id = accum_torch[batch_id]
            batch_id_torch_cuda[start_id:end_id, 0] = batch_id
            start_id = end_id

        coors = torch.cat([batch_id_torch_cuda, coors], dim=1)

        batch_dict['point_features'] = features
        batch_dict['point_coords'] = coors
        return batch_dict




class DynamicPointConvConcatBackBone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        '''
        BACKBONE_3D:
            NAME: DynamicPointConvBackBone
            LAYER_PARAMS:
                PADDING: 0.0
                SUBSAMPLE_RES: [0.1, 0.2, 0.4, 0.8, 1.6]
                OUTPUT_CHANNELS: [16, 32, 64, 128, 256]
                KERNEL_RES: [0.2, 0.4, 0.8, 1.6, 3.2]
                KERNEL_SIZE: [3, 3, 3, 3, 3]
                HEADS: [False, False, False, False, True] # new
            MODEL_PARAMS:
                ACTIVATION: 'relu'
                GRID_BUFFER_SIZE: 3
                OUTPUT_POOLING_SIZE: 5
                SAVE_MEM: true
            DIMENSION_PARAMS:
                DIMENSIONS: null # need to modified to point cloud range here if possible
                COOR_OFFSET: null
 
        '''
        
        self.dvp_conv_layers = nn.ModuleList()

        output_channels = self.model_cfg.LAYER_PARAMS.OUTPUT_CHANNELS.copy()
        subsample_res = self.model_cfg.LAYER_PARAMS.SUBSAMPLE_RES.copy()
        kernel_res = self.model_cfg.LAYER_PARAMS.KERNEL_RES.copy()
        heads = self.model_cfg.LAYER_PARAMS.HEADS.copy()
        kernel_size = self.model_cfg.LAYER_PARAMS.KERNEL_SIZE.copy()
        self.mem_saving = self.model_cfg.MODEL_PARAMS.SAVE_MEM
        self.num_keypoints = self.model_cfg.MODEL_PARAMS.NUM_KEYPOINTS
        # print("kernel_size: ", kernel_size)
        # print("self.model_cfg.LAYER_PARAMS.PADDING: ", self.model_cfg.LAYER_PARAMS.PADDING)

        for k in range(self.model_cfg.LAYER_PARAMS.SUBSAMPLE_RES.__len__()):
            # print("K Layer: ", k )
            self.dvp_conv_layers.append(
                DynamicPointConvBasicBlock(input_channels=output_channels[k],
                        output_channels=output_channels[k+1],
                        grid_buffer_size=self.model_cfg.MODEL_PARAMS.GRID_BUFFER_SIZE,
                        output_pooling_size=self.model_cfg.MODEL_PARAMS.OUTPUT_POOLING_SIZE,
                        subsample_res=subsample_res[k],
                        kernel_res=kernel_res[k],
                        dimension=self.model_cfg.DIMENSION_PARAMS.DIMENSIONS,
                        coor_offset=self.model_cfg.DIMENSION_PARAMS.COOR_OFFSET,
                        activation=self.model_cfg.MODEL_PARAMS.ACTIVATION,
                        kernel_size=kernel_size[k],
                        padding=self.model_cfg.LAYER_PARAMS.PADDING,
                        bn_decay=None,
                        last_layer=heads[k],
                        concat_feature=True)
            )
        self.num_point_features = np.sum(output_channels)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        num_points_list = batch_dict['num_points']
        coors, features, num_list = points[:,1:4].contiguous(), points[:,4:].contiguous(), num_points_list
        concat_features = features
        voxel_idx, center_idx = None, None
        
        for i in range(len(self.dvp_conv_layers)):
            coors, features, num_list, voxel_idx, center_idx, concat_features = \
                self.dvp_conv_layers[i](coors,
                                        features,
                                        num_list,
                                        voxel_idx,
                                        center_idx,
                                        mem_saving=self.mem_saving,
                                        concat_features=concat_features)
            # print(coors.shape, features.shape, num_list.shape, voxel_idx.shape, center_idx.shape)
        
        # add batch idx to point_coors
        num_sampled_points = torch.tensor(self.num_keypoints, 
            dtype=torch.int32, device=coors.device)
        # batch_id_torch_cuda = torch.zeros((coors.shape[0], 1), 
        #     dtype=torch.float32,
        #     device=coors.device)
        
        accum_torch = torch.cumsum(num_list, dim=0).type(torch.int64)
        resampled_coor = torch.zeros((num_sampled_points * batch_size, 4), 
            dtype=torch.float32,
            device=coors.device)
        resampled_features = torch.zeros((num_sampled_points *batch_size, concat_features.shape[1]), 
            dtype=torch.float32,
            device=concat_features.device)
        
        start_idx = 0
        end_idx = torch.min(num_list[0] , num_sampled_points)

        resampled_coor[:end_idx, 1:] = coors[:end_idx, :]
        resampled_features[:end_idx, :] = concat_features[:end_idx, :]

        for batch_id in range(1, batch_size):
            start_idx = batch_id*num_sampled_points
            end_idx = torch.min(num_list[batch_id] , num_sampled_points)
            # end_idx = np.min([accum_torch[batch_id].detach().cpu().numpy() , num_sampled_points])

            resampled_coor[start_idx: start_idx+end_idx, 1:] = \
                coors[accum_torch[batch_id-1]: (accum_torch[batch_id-1]+end_idx), :]

            resampled_features[start_idx: start_idx+end_idx,:] = \
                concat_features[accum_torch[batch_id-1]: (accum_torch[batch_id-1]+end_idx), :]

            # resampled_coor[start_idx: start_idx+end_idx, 0] = batch_id
            resampled_coor[(batch_id-1)*num_sampled_points:batch_id*num_sampled_points, 0] = batch_id
            # resample_points
            # batch_id_torch_cuda[(batch_id-1)*num_sampled_points:batch_id*num_sampled_points, 0] = batch_id

        # print("num_list: ", num_list)
        # print("resampled_features.shape: ", resampled_features.shape, " self.num_point_features: ", self.num_point_features)
        batch_dict['point_features'] = resampled_features
        batch_dict['point_coords'] = resampled_coor
        # print(torch.min(resampled_coor, axis=0))
        # print(batch_id_torch_cuda)
        # batch_dict['point_features'] = features
        # batch_dict['point_coords'] = torch.cat((batch_id_torch_cuda, coors), dim=1)
        # print("batch_dict['point_features'].shape: ", batch_dict['point_features'].shape)
        # print("batch_dict['point_coords'].shape: ", batch_dict['point_coords'].shape)
        # print("batch_dict['point_coords']: ", batch_dict['point_coords'][0,:])
        return batch_dict