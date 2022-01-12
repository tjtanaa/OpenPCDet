import torch
import torch.nn as nn
from torch.autograd import Function

from . import grid_sampling_cuda
from . import voxel_sampling_feature_cuda
from . import voxel_sampling_idx_binary_cuda
from . import voxel_sampling_idx_cuda

def grid_sampling_gpu(input_coors,
                  input_num_list,
                  resolution,
                  dimension=None,
                  coor_offset=None):
    """
    :param input_coors: (M, 3), float
    :param input_num_list: (B), num_points_in_each_batch, int
    :param resolution: (1), float
    :return 

    """
    assert input_coors.dim() == 2
    num_points = input_coors.shape[0]
    batch_size = input_num_list.size(0)

    # print(dimension)
    # print(coor_offset)
    if (dimension is None and coor_offset is None):
        min_coors, _ = torch.min(input_coors, axis=0)
        max_coors, _ = torch.max(input_coors, axis=0)
        # min_coors = min_coors - 0.5
        # max_coors = max_coors + 0.5
        dimension = max_coors - min_coors
        coor_offset = -min_coors
        # print(min_coors)

    dimension = torch.tensor(dimension, dtype=torch.float32, device=input_coors.device, requires_grad=False) #.to(input_coors.device).type(torch.float32)
    coor_offset = torch.tensor(coor_offset, dtype=torch.float32, device=input_coors.device, requires_grad=False) #.to(input_coors.device).type(torch.float32)


    # min_offset = torch.tensor([-180, -140, -1.5] ).to('cuda')
    # dimension = torch.tensor([185, 160, 7.0]).to('cuda')
    grid_dimension = torch.ceil(dimension / resolution).type(torch.int32)

    # print(coor_offset)
    # # print(input_num_list)
    # print(grid_dimension)
    # min_offset_coors, _ = torch.min(input_coors + coor_offset, axis=0)
    # max_offset_coors, _ = torch.max(input_coors + coor_offset, axis=0) 
    # print(min_offset_coors)
    # print(max_offset_coors)

    output_idx, output_num_list = grid_sampling_cuda.grid_sampling_gpu(
        int(grid_dimension[0].detach().cpu()), 
        int(grid_dimension[1].detach().cpu()), 
        int(grid_dimension[2].detach().cpu()),
        resolution,
        resolution,
        resolution,
        (input_coors + coor_offset).contiguous(), 
        input_num_list.contiguous()
        )


    output_coors = input_coors[output_idx.type(torch.int64),:]

    return output_coors, output_num_list, output_idx

def voxel_sampling_idx_binary_gpu(input_coors,
                              input_features,
                              input_num_list,
                              center_coors,
                              center_num_list,
                              resolution,
                              dimension=None,
                              coor_offset=None,
                              kernel_size=3,
                              grid_buffer_size=3,
                              output_pooling_size=5):

    npoint = input_coors.size()[0]
    batch_size = input_num_list.size()[0]
    
    if (dimension is None and coor_offset is None):
        min_coors, _ = torch.min(input_coors, axis=0)
        max_coors, _ = torch.max(input_coors, axis=0)
        # min_coors = min_coors - 0.5
        # max_coors = max_coors + 0.5
        dimension = max_coors - min_coors
        coor_offset = -min_coors

    dimension = torch.tensor(dimension, dtype=torch.float32, device=input_coors.device, requires_grad=False) #.to(input_coors.device).type(torch.float32)
    coor_offset = torch.tensor(coor_offset, dtype=torch.float32, device=input_coors.device, requires_grad=False) #.to(input_coors.device).type(torch.float32)

    grid_dimension = torch.ceil(dimension / resolution).type(torch.int64)

    dim_w = grid_dimension[0]
    dim_l = grid_dimension[1]
    dim_h = grid_dimension[2]

    dim_offset = torch.prod(grid_dimension)

    # print("npoint: ", npoint)
    # print("input_coors.get_device(): ", input_coors.device)
    point_ids = torch.arange(npoint).to(input_coors.device) + 1
    point_ids_array = torch.unsqueeze(point_ids, dim=0).repeat(batch_size, 1).type(torch.float32)
    accu_num_list = torch.cumsum(input_num_list, dim=0).type(torch.float32)
    masks = torch.gt(point_ids_array / torch.unsqueeze(accu_num_list, axis=-1), 1.0).type(torch.int64)
    voxel_offset_masks = torch.sum(masks, 0) * dim_offset

    input_voxel_coors = torch.floor((input_coors + coor_offset) / resolution).type(torch.int64)
    # input_voxel_coors = tf.clip_by_value(input_voxel_coors, clip_value_min=0, clip_value_max=[dim_w - 1, dim_l - 1, dim_h - 1])
    input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    input_voxel_ids += voxel_offset_masks
    sorted_args = torch.argsort(input_voxel_ids)
    # print(input_voxel_coors.shape)
    # print(sorted_args.shape)
    sorted_voxel_ids = input_voxel_ids[sorted_args] - voxel_offset_masks
    sorted_coors = input_coors[sorted_args, :]
    sorted_features = input_features[sorted_args,:]
    # XXX: Need to pay attention to the back-propagation implementation.
    output_idx = voxel_sampling_idx_binary_cuda.voxel_sampling_idx_binary_gpu(
                                                                kernel_size,
                                                                dimension[0], dimension[1], dimension[2],
                                                                grid_dimension[0], grid_dimension[1], grid_dimension[2],
                                                                resolution, resolution, resolution,
                                                                grid_buffer_size,
                                                                output_pooling_size,
                                                                # dimension,
                                                                sorted_coors + coor_offset,
                                                                sorted_voxel_ids.type(torch.int64),
                                                                input_num_list,
                                                                center_coors + coor_offset,
                                                                center_num_list)

                                                                
    return output_idx, sorted_features

def voxel_sampling_idx_gpu(input_coors,
                       input_features,
                       input_num_list,
                       center_coors,
                       center_num_list,
                       resolution,
                       dimension=None,
                       coor_offset=None,
                       kernel_size=3,
                       grid_buffer_size=3,
                       output_pooling_size=5):

    
    # min_offset, _ = torch.min(input_coors, axis=0)
    # max_offset, _ = torch.max(input_coors, axis=0)
    # dimension = max_offset - min_offset

    if (dimension is None and coor_offset is None):
        min_coors, _ = torch.min(input_coors, axis=0)
        max_coors, _ = torch.max(input_coors, axis=0)        
        # min_coors = min_coors - 0.5
        # max_coors = max_coors + 0.5
        dimension = max_coors - min_coors
        coor_offset = -min_coors

    dimension = torch.tensor(dimension, dtype=torch.float32, device=input_coors.device, requires_grad=False) #.to(input_coors.device).type(torch.float32)
    coor_offset = torch.tensor(coor_offset, dtype=torch.float32, device=input_coors.device, requires_grad=False) #.to(input_coors.device).type(torch.float32)

    grid_dimension = torch.ceil(dimension / resolution).type(torch.int32)
    
    output_idx = voxel_sampling_idx_cuda.voxel_sampling_idx_gpu(resolution,
                                                                kernel_size,
                                                                dimension[0], dimension[1], dimension[2],
                                                                grid_dimension[0], grid_dimension[1], grid_dimension[2],
                                                                grid_buffer_size,
                                                                output_pooling_size,
                                                                input_coors + coor_offset,
                                                                input_num_list,
                                                                center_coors + coor_offset,
                                                                center_num_list)
    return output_idx, input_features

def voxel_sampling_feature_gpu(input_features,
                           output_idx,
                           kernel_size=3,
                           padding=-1.0):
    # print("kernel_szie: ", kernel_size, "\t padding: ", padding)
    output_features = voxel_sampling_feature_cuda.voxel_sampling_feature_gpu(padding,
                                            kernel_size,
                                            input_features,
                                            output_idx
                                            )
    return output_features

# to be called in the nn.Module
def voxel_sampling_feature_grad_gpu(input_features, output_idx, grad):
    input_features_grad = voxel_sampling_feature_cuda.voxel_sampling_feature_grad_gpu(input_features,
                                                                        output_idx,
                                                                        grad)
    return input_features_grad



class VoxelSamplingFeatureFunction(Function):
    @staticmethod
    def forward(ctx, input_features,
                           output_idx,
                           kernel_size=3,
                           padding=-1.0):
        """
        Args:
            ctx:
            input_features: (npoints, C)

        Returns:
            
        """
        # assert rois.shape[1] == 7 and pts.shape[1] == 3
        # if isinstance(out_size, int):
        #     out_x = out_y = out_z = out_size
        # else:
        #     assert len(out_size) == 3
        #     for k in range(3):
        #         assert isinstance(out_size[k], int)
        #     out_x, out_y, out_z = out_size
        
        input_point_num = input_features.shape[0]
        channels = input_features.shape[1]

        output_features = voxel_sampling_feature_gpu(input_features=input_features,
                           output_idx=output_idx,
                           kernel_size=kernel_size,
                           padding=padding)

        ctx.voxel_sampling_feature_for_backward = (input_features,
                                                    output_idx,
                                                    kernel_size,
                                                    padding, input_point_num, channels)
        return output_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        input_features, output_idx, kernel_size, padding, input_point_num, channels = ctx.voxel_sampling_feature_for_backward

        # grad_in = grad_out.new_zeros((input_point_num, channels))

        grad_in = voxel_sampling_feature_grad_gpu(input_features, output_idx, grad_out)
        # print("grad_in: ", grad_in)
        return grad_in, None, None, None

if __name__ == '__main__':
    pass

    print("Test Case: grid_sampling_gpu")

    import numpy as np
    from tqdm import tqdm
    import timeit
    from torch.profiler import profile, record_function, ProfilerActivity
    from point_viz.converter import PointvizConverter

    Converter = PointvizConverter("/media/data2/dvpdet_visualization")

    def plot_points_from_voxels_with_color(voxels, center_coors, resolution, self_rgbs=False, kernel_size=3, mask=-1, name='test'):
        output_coors = []
        output_rgb = []
        half_kernel_size = (kernel_size - 1) / 2
        for i in tqdm(range(len(voxels))):
            r, g, b = np.random.randint(low=0, high=255, size=3)
            for n in range(kernel_size ** 3):
                intensity = voxels[i, n, 0]
                if intensity != mask:
                    x = n % kernel_size
                    z = n // (kernel_size ** 2)
                    y = (n - z * kernel_size ** 2) // kernel_size
                    x_coor = (x - half_kernel_size) * resolution + center_coors[i, 0]
                    y_coor = (y - half_kernel_size) * resolution + center_coors[i, 1]
                    z_coor = (z - half_kernel_size) * resolution + center_coors[i, 2]
                    output_coors.append([x_coor, y_coor, z_coor])
                    if not self_rgbs:
                        output_rgb.append([r, g, b])
                    else:
                        output_rgb.append(voxels[i, n, :])

        output_coors, output_rgb = np.array(output_coors), np.array(output_rgb)
        Converter.compile(coors=output_coors[:,[1,2,0]],
                        default_rgb=output_rgb,
                        task_name=name)

    def fetch_instance(input_list, num_list, id=0):
        accu_num_list = np.cumsum(num_list)
        if id == 0:
            return input_list[:num_list[0], ...]
        else:
            return input_list[accu_num_list[id - 1]:accu_num_list[id], ...]

    # numpoints
    batch_size = 4
    num_points = 1000
    vmin = 0
    vmax = 2
    sphere_np = np.random.uniform(vmin, vmax, size=(batch_size * num_points,3)).astype(np.float32)
    sphere_np[0,:] = vmin
    sphere_np[-1,:] = vmax

    sphere_torch_cuda = torch.from_numpy(sphere_np).to('cuda')

    input_num_list_np = np.array([num_points for i in range(batch_size)]).astype(np.int32)
    resolution = 0.1
    input_num_list_torch_cuda = torch.from_numpy(input_num_list_np).to('cuda')

    # grid_sampling_gpu

    output_coors, output_num_list, output_idx = \
        grid_sampling_gpu(sphere_torch_cuda,
                                        input_num_list_torch_cuda,
                                        resolution)

    print(output_coors.shape)
    print(output_num_list.shape)
    print(output_num_list.sum())
    print(output_idx.shape)

    def get_rgbs_from_coors_torch(coors, repeat=5):
        norm_coors = coors - torch.min(coors, axis=0, keepdims=True)[0]
        norm_coors = norm_coors / torch.max(norm_coors, axis=0, keepdims=True)[0]
        return norm_coors * repeat * 255 % 255.


    voxel_idx, features = voxel_sampling_idx_binary_gpu(input_coors=sphere_torch_cuda,
                                             input_features=get_rgbs_from_coors_torch(sphere_torch_cuda),
                                             input_num_list=input_num_list_torch_cuda,
                                             center_coors=output_coors,
                                             center_num_list=output_num_list,
                                             resolution=0.2,
                                             kernel_size=3,
                                             grid_buffer_size=3,
                                             output_pooling_size=5)

    print(voxel_idx.shape)


    voxels = voxel_sampling_feature_gpu(input_features=features,
                                output_idx=voxel_idx,
                                padding=-1)
    print(voxels.shape)
    print(voxels[0,:,:])

    print("Test Case 2: Benchmark")
    epoch= 100
    # coors, features, num_list, voxels = point_sampling(coors, features, num_list, 16,0.8, 'layer_0')
    offset_training = [180, 140, 1.5] 
    dimension_training = [185, 160, 7.0]


    # grid_sampling_gpu
    # start = timeit.timeit()

    grid_buffer_size = 3
    output_pooling_size = 5
    
    input_coors_batch_np = np.load("test/input_coors.npy", allow_pickle=True)
    # input_features = np.load("input_features.npy", allow_pickle=True)
    input_num_list_batch_np = np.load("test/input_num_list.npy", allow_pickle=True)
    # output_voxels_0_np = np.load("test/output_voxels.npy", allow_pickle=True)

    # input_coors_batch_torch_cuda = [torch.from_numpy(input_coors_batch_np[i].astype(np.float32)).to('cuda').contiguous() for i in range(epoch)]
    # input_num_list_batch_torch_cuda = [torch.from_numpy(input_num_list_batch_np[i].astype(np.int32)).to('cuda').contiguous() for i in range(epoch)]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(
    #         wait=1,
    #         warmup=1,
    #         active=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/dvp_conv_5')
    # ) as p:
    for i in tqdm(range(epoch)):
        sample_idx = i % 100
        # input_coors_torch_cuda = input_coors_batch_torch_cuda[i]
        # input_num_list_torch_cuda = input_num_list_batch_torch_cuda[i]
        input_coors_torch_cuda = torch.from_numpy(input_coors_batch_np[sample_idx].astype(np.float32)).to('cuda').contiguous()
        input_num_list_torch_cuda = torch.from_numpy(input_num_list_batch_np[sample_idx].astype(np.int32)).to('cuda').contiguous()

        coors_0, num_list_0, _ = \
            grid_sampling_gpu(input_coors_torch_cuda,
                                input_num_list_torch_cuda,
                                0.1)
                                
        # voxel_idx, features = voxel_sampling_idx_binary_gpu(input_coors=input_coors_torch_cuda,
        #                                         input_features=get_rgbs_from_coors_torch(input_coors_torch_cuda),
        #                                         input_num_list=input_num_list_torch_cuda,
        #                                         center_coors=coors_0,
        #                                         center_num_list=num_list_0,
        #                                         resolution=0.2,
        #                                         grid_buffer_size=3,
        #                                         output_pooling_size=5)

        voxel_idx, features = voxel_sampling_idx_gpu(input_coors=input_coors_torch_cuda,
                                                input_features=get_rgbs_from_coors_torch(input_coors_torch_cuda),
                                                input_num_list=input_num_list_torch_cuda,
                                                center_coors=coors_0,
                                                center_num_list=num_list_0,
                                                resolution=0.2,
                                                grid_buffer_size=3,
                                                output_pooling_size=5)

        voxels = voxel_sampling_feature_gpu(input_features=features,
                                    output_idx=voxel_idx,
                                    padding=-1)
        if i == 0:
            id = 0
            output_voxels = fetch_instance(voxels.detach().cpu().numpy(), num_list_0.detach().cpu().numpy(), id=id)
            output_centers = fetch_instance(coors_0.detach().cpu().numpy(), num_list_0.detach().cpu().numpy(), id=id)

            plot_points_from_voxels_with_color(voxels=output_voxels,
                                               center_coors=output_centers,
                                               resolution=0.2,
                                               self_rgbs=True,
                                               name='voxel_sampling')
            #
        #     pass
        #     print(input_num_list_torch_cuda,  voxels.shape)
            # print(i, input_num_list_torch_cuda,  coors_0.shape, num_list_0, torch.sum(num_list_0))
            # voxels_np = voxels.detach().cpu().numpy()
            # mask = np.abs(output_voxels_0_np - voxels_np) > 1e-5
            # print("number of different pixel: ", np.sum(mask.astype(np.int32)))
        # p.step()

    # end = timeit.timeit()
    # print((end-start) / epoch)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end)/ 1000 / epoch)