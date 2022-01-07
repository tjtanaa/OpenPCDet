/* Voxel Sampling Operation
 * with unstructured number of input points for each mini batch
 * Created by Zhaoyu SU
 * Ported by Tun Jian TAN
 * All Rights Reserved. Sep., 2019.
 */
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda.h>
#include <math.h>
#include <assert.h>


// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void voxel_sampling_idx_binary_gpu_launcher(int batch_size, int input_npoint,
                                            int center_num, int kernel_size,
                                            float dim_w, float dim_l, float dim_h,
                                            int grid_w, int grid_l, int grid_h,
                                            float res_x, float res_y, float res_z,
                                            int grid_buffer_size, int output_pooling_size,
                                            const float* input_coors,
                                            const int64_t* input_voxel_idx,
                                            const int* input_num_list,
                                            const float* center_coors,
                                            const int* center_num_list,
                                            int* input_accu_list,
                                            int* center_accu_list,
                                            int* output_idx,
                                            int* output_idx_count);

torch::Tensor voxel_sampling_idx_binary_gpu(
    int kernel_size,
    float dim_w, float dim_l, float dim_h,
    int grid_w, int grid_l, int grid_h,
    float res_x, float res_y, float res_z,
    int grid_buffer_size, 
    int output_pooling_size,
    torch::Tensor input_coors, 
    torch::Tensor input_voxel_idx,
    torch::Tensor input_num_list,
    torch::Tensor center_coors,
    torch::Tensor center_num_list
){

    CHECK_INPUT(input_coors);
    CHECK_INPUT(input_voxel_idx);
    CHECK_INPUT(input_num_list);
    CHECK_INPUT(center_coors);
    CHECK_INPUT(center_num_list);


    const float * input_coors_ptr = input_coors.data_ptr<float>();
    const int64_t * input_voxel_idx_ptr = input_voxel_idx.data_ptr<int64_t>();
    const int * input_num_list_ptr = input_num_list.data_ptr<int>();
    const float * center_coors_ptr = center_coors.data_ptr<float>();
    const int * center_num_list_ptr = center_num_list.data_ptr<int>();


    int input_npoint = input_coors.size(0);
    int center_num = center_coors.size(0);
    int batch_size = input_num_list.size(0);
    int kernel_num = kernel_size * kernel_size * kernel_size;

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
//            printf("%d=%d+%d, %d\n", center_accu_list_ptr_host[b], center_accu_list_ptr_host[b-1], center_num_list_ptr_host[b-1], center_num_list_ptr_host[b]);
    }

    // int count = 0;
    // for (int b=0; b<batch_size; b++) {
    //     count += center_num_list_ptr_host[b];
    // }
    // if (count != center_num)
    //     printf("*****************Mismatch Dimension: %d vs. %d; input_npoint=%d\n", center_num, count, input_npoint);


    torch::Tensor input_accu_list = torch::zeros({batch_size}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * input_accu_list_ptr = input_accu_list.data_ptr<int>();
    cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);


    torch::Tensor center_accu_list = torch::zeros({batch_size}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * center_accu_list_ptr = center_accu_list.data_ptr<int>();
    cudaMemcpy(center_accu_list_ptr, center_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);

    torch::Tensor output_idx = torch::zeros(
        {center_num, kernel_num, output_pooling_size}, 
        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * output_idx_ptr = output_idx.data_ptr<int>();

    
    torch::Tensor output_idx_count = torch::zeros(
        {center_num * kernel_num}, 
        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * output_idx_count_ptr = output_idx_count.data_ptr<int>();

    voxel_sampling_idx_binary_gpu_launcher(batch_size, input_npoint,
                                            center_num, kernel_size,
                                            dim_w, dim_l, dim_h,
                                            grid_w, grid_l, grid_h,
                                            res_x, res_y, res_z,
                                            grid_buffer_size, output_pooling_size,
                                            input_coors_ptr,
                                            input_voxel_idx_ptr,
                                            input_num_list_ptr,
                                            center_coors_ptr,
                                            center_num_list_ptr,
                                            input_accu_list_ptr,
                                            center_accu_list_ptr,
                                            output_idx_ptr,
                                            output_idx_count_ptr);

    free(input_num_list_ptr_host);
    free(center_num_list_ptr_host);
    free(input_accu_list_ptr_host);
    free(center_accu_list_ptr_host);

    return output_idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxel_sampling_idx_binary_gpu", &voxel_sampling_idx_binary_gpu, "voxel_sampling_idx_binary_gpu forward (CUDA)");
}
