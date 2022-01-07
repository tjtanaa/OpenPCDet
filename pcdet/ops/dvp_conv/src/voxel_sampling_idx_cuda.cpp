/* Voxel Sampling Operation
 * with unstructured number of input points for each mini batch
 * Created by Zhaoyu SU
 * Ported by Tun Jian TAN
 * All Rights Reserved. Sep., 2019.
 */
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <climits>

// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void voxel_sampling_idx_gpu_launcher(int batch_size, int input_point_num,
                                     int center_num, int kernel_size,
                                     int grid_dim_w, int grid_dim_l, int grid_dim_h,
                                     float resolution,
                                     int grid_buffer_size, int output_pooling_size,
                                     const float* input_coors,
                                     const int* input_num_list,
                                     const float* center_coors,
                                     const int* center_num_list,
                                     int* input_accu_list,
                                     int* center_accu_list,
                                     int* grid_buffer,
                                     int* grid_buffer_count,
                                     int* output_idx,
                                     int* output_idx_count);

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
){

    CHECK_INPUT(input_coors);
    CHECK_INPUT(input_num_list);
    CHECK_INPUT(center_coors);
    CHECK_INPUT(center_num_list);
    
    const float * input_coors_ptr = input_coors.data_ptr<float>();
    const int * input_num_list_ptr = input_num_list.data_ptr<int>();
    const float * center_coors_ptr = center_coors.data_ptr<float>();
    const int * center_num_list_ptr = center_num_list.data_ptr<int>();

    int input_point_num = input_coors.size(0);
    int center_num = center_coors.size(0);
    int batch_size = input_num_list.size(0);
    int kernel_num = kernel_size * kernel_size * kernel_size;
    // if (INT_MAX / grid_dim_h / grid_dim_l / grid_dim_w < batch_size){
    //     printf("VoxelSamplingOp ERROR: size of grid buffer %d x [%d x %d x %d] exceeds INT32 range: %d.\n",
    //             batch_size, grid_dim_w, grid_dim_l, grid_dim_h, INT_MAX);}

//        printf("******************input shape = %d************************\n", input_point_num);
//        printf("******************output shape = %d************************\n", kernel_number);



    int batch_byte_size = batch_size * sizeof(int);
    int* input_num_list_ptr_host = (int*)malloc(batch_byte_size);
    int* center_num_list_ptr_host = (int*)malloc(batch_byte_size);
    int* input_accu_list_ptr_host = (int*)malloc(batch_byte_size);
    int* center_accu_list_ptr_host = (int*)malloc(batch_byte_size);
    input_accu_list_ptr_host[0] = 0;
    center_accu_list_ptr_host[0] = 0;
    cudaMemcpy(input_num_list_ptr_host, input_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(center_num_list_ptr_host, center_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);

    for (int b=1; b<batch_size; b++) {
        input_accu_list_ptr_host[b] = input_accu_list_ptr_host[b-1] + input_num_list_ptr_host[b-1];
        center_accu_list_ptr_host[b] = center_accu_list_ptr_host[b-1] + center_num_list_ptr_host[b-1];
    }

    torch::Tensor input_accu_list = torch::zeros({batch_size}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * input_accu_list_ptr = input_accu_list.data_ptr<int>();
    cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);


    torch::Tensor center_accu_list = torch::zeros({batch_size}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * center_accu_list_ptr = center_accu_list.data_ptr<int>();
    cudaMemcpy(center_accu_list_ptr, center_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);


    torch::Tensor grid_buffer = torch::full({batch_size, grid_dim_w, grid_dim_l, grid_dim_h, grid_buffer_size}, 0xEF,
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * grid_buffer_ptr = grid_buffer.data_ptr<int>();


    torch::Tensor grid_buffer_count = torch::zeros({batch_size, grid_dim_w, grid_dim_l, grid_dim_h}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * grid_buffer_count_ptr = grid_buffer_count.data_ptr<int>();

    torch::Tensor output_idx = torch::zeros(
        {center_num, kernel_num, output_pooling_size}, 
        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * output_idx_ptr = output_idx.data_ptr<int>();

    
    torch::Tensor output_idx_count = torch::zeros(
        {center_num * kernel_num}, 
        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * output_idx_count_ptr = output_idx_count.data_ptr<int>();


    voxel_sampling_idx_gpu_launcher(batch_size, input_point_num,
                                    center_num, kernel_size,
                                    grid_dim_w, grid_dim_l, grid_dim_h,
                                    resolution,
                                    grid_buffer_size, output_pooling_size,
                                    input_coors_ptr,
                                    input_num_list_ptr,
                                    center_coors_ptr,
                                    center_num_list_ptr,
                                    input_accu_list_ptr,
                                    center_accu_list_ptr,
                                    grid_buffer_ptr,
                                    grid_buffer_count_ptr,
                                    output_idx_ptr,
                                    output_idx_count_ptr);

    free(input_num_list_ptr_host);
    free(center_num_list_ptr_host);

    return output_idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxel_sampling_idx_gpu", &voxel_sampling_idx_gpu, "voxel_sampling_idx_gpu forward (CUDA)");
}
