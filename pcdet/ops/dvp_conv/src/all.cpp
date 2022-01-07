/* 
 * Created by Tun Jian TAN
 * All Rights Reserved. Sep., 2021.
 */
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>

/**
 * Grid down-sampling point cloud
 *
 * From grid_sampling_cuda.cpp and grid_sampling_cuda_kernel.cu
 * 
 * @param resolution. Resolution of the grid.
 * @param grid_w. The grid size along x-axis
 * @param grid_l. The grid size along y-axis
 * @param grid_h. The grid size along z-axis
 * @return {output_idx, output_num_list}. vector{IntTensor, IntTensor(B)}
 */
std::vector<torch::Tensor> grid_sampling_gpu(
    float resolution, 
    int grid_w, int grid_l, int grid_h,
    torch::Tensor input_coors, torch::Tensor input_num_list 
    );

/**
 * Compute the output idx of voxel sampling
 *
 * From voxel_sampling_idx_binary_cuda.cpp and voxel_sampling_idx_binary_cuda_kernel.cu
 * 
 * @param resolution. Resolution of the grid.
 * @param kernel_size. Size of convolutional kernel.
 * @param dim_w. The dimension/range (float) along x-axis
 * @param dim_l. The dimension/range (float) along y-axis
 * @param dim_h. The dimension/range (float) along z-axis
 * @param grid_buffer_size. Number of points to be kept in a grid/voxel.
 * @param output_pooling_size. Number of points to be kept to be multiplied with the kernel weights
 * @param input_coors. FloatTensor(N, 3) input point cloud coordinates [x,y,z]
 * @param input_voxel_idx. IntTensor(N), voxel index of point
 * @param input_num_list. IntTensor(B), Number of points in each batch. 
 * @param center_coors. FloatTensor(M, 3). Points from original point cloud that is selected to be center of a voxel
 * @param center_num_list. IntTensor(B). Number of center points in each batch.
 * @return output_idx. IntTensor
 */
torch::Tensor voxel_sampling_idx_binary_gpu(
    float resolution, 
    int kernel_size,
    float dim_w, float dim_l, float dim_h,
    int grid_buffer_size, 
    int output_pooling_size,
    torch::Tensor input_coors, 
    torch::Tensor input_voxel_idx,
    torch::Tensor input_num_list,
    torch::Tensor center_coors,
    torch::Tensor center_num_list
);


/**
 * Compute the output idx of voxel sampling
 *
 * From voxel_sampling_idx_binary_cuda.cpp and voxel_sampling_idx_binary_cuda_kernel.cu
 * 
 * @param resolution. Resolution of the grid.
 * @param kernel_size. Size of convolutional kernel.
 * @param dim_w. The dimension/range (float) along x-axis
 * @param dim_l. The dimension/range (float) along y-axis
 * @param dim_h. The dimension/range (float) along z-axis
 * @param grid_dim_w. The number of voxels (float) along x-axis
 * @param grid_dim_l. The number of voxels (float) along y-axis
 * @param grid_dim_h. The number of voxels (float) along z-axis
 * @param grid_buffer_size. Number of points to be kept in a grid/voxel.
 * @param output_pooling_size. Number of points to be kept to be multiplied with the kernel weights
 * @param input_coors. FloatTensor(N, 3) input point cloud coordinates [x,y,z]
 * @param input_num_list. IntTensor(B), Number of points in each batch. 
 * @param center_coors. FloatTensor(M, 3). Points from original point cloud that is selected to be center of a voxel
 * @param center_num_list. IntTensor(B). Number of center points in each batch.
 * @return output_idx. IntTensor
 */
torch::Tensor voxel_sampling_idx_gpu(
    float resolution, 
    int kernel_size,
    float dim_w, float dim_l, float dim_h,
    int grid_dim_w, int grid_dim_l, int grid_dim_h,
    int grid_buffer_size, 
    int output_pooling_size,
    torch::Tensor input_coors, 
    torch::Tensor input_num_list,
    torch::Tensor center_coors,
    torch::Tensor center_num_list
);

/**
 * Sample voxel features from output index
 *
 * From voxel_sampling_feature_cuda.cpp and voxel_sampling_feature_cuda_kernel.cu
 * 
 * @param padding_value. Default values of point features if not enough point.
 * @param kernel_size. Size of convolutional kernel.
 * @param input_features. FloatTensor(N, C). Point Features
 * @param output_idx. IntTensor(M). Selected indexes of output points.
 * @return output_features. FloatTensor
 */
torch::Tensor voxel_sampling_feature_gpu(
    float padding_value,
    int kernel_size,
    torch::Tensor input_features,
    torch::Tensor output_idx
);

/**
 * Compute gradient of input features of voxel sampling feature Op
 *
 * @param padding_value. Default values of point features if not enough point.
 * @param kernel_size. Size of convolutional kernel.
 * @param input_features. FloatTensor(N, C). Point Features
 * @param output_idx. IntTensor(M). Selected indexes of output points.
 * @param output_features_grad. FloatTensor. Gradients of output features.
 * @return input_features_grad. FloatTensor
 */
torch::Tensor voxel_sampling_feature_grad_gpu(
    torch::Tensor input_features,
    torch::Tensor output_idx,
    torch::Tensor output_features_grad
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_sampling_gpu", &grid_sampling_gpu, "grid_sampling_gpu forward (CUDA)");
    m.def("voxel_sampling_idx_binary_gpu", &voxel_sampling_idx_binary_gpu, "voxel_sampling_idx_binary_gpu forward (CUDA)");
    m.def("voxel_sampling_idx_gpu", &voxel_sampling_idx_gpu, "voxel_sampling_idx_gpu forward (CUDA)");
    m.def("voxel_sampling_feature_gpu", &voxel_sampling_feature_gpu, "voxel_sampling_feature_gpu forward (CUDA)");
    m.def("voxel_sampling_feature_grad_gpu", &voxel_sampling_feature_grad_gpu, "voxel_sampling_feature_gpu backward (CUDA)");
}
