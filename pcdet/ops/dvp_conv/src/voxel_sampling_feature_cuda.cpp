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
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>


// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void voxel_sampling_feature_gpu_launcher(int center_num, int kernel_num, int channels, float padding,
                                         int output_pooling_size,
                                         const float* input_features,
                                         const int* output_idx,
                                         float* output_features);

torch::Tensor voxel_sampling_feature_gpu(
    float padding_value,
    int kernel_size,
    torch::Tensor input_features,
    torch::Tensor output_idx
){
    CHECK_INPUT(input_features);
    CHECK_INPUT(output_idx);

    const float * input_features_ptr = input_features.data_ptr<float>();
    const int * output_idx_ptr = output_idx.data_ptr<int>();

    int channels = input_features.size(1);
    int output_pooling_size = output_idx.size(2);
    int kernel_num = kernel_size * kernel_size * kernel_size;
    int center_num = output_idx.size(0);

//        printf("******************input shape = %d************************\n", input_point_num);
//        printf("******************output shape = %d************************\n", kernel_num);


    torch::Tensor output_features = torch::zeros(
        {center_num, kernel_num, channels}, 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(input_features.device().type(), input_features.device().index()).requires_grad(true));
    
    float * output_features_ptr = output_features.data_ptr<float>();


    voxel_sampling_feature_gpu_launcher(center_num, kernel_num, channels, padding_value,
                                        output_pooling_size,
                                        input_features_ptr,
                                        output_idx_ptr,
                                        output_features_ptr);
    return output_features;
}

void voxel_sampling_feature_grad_gpu_launcher(int center_num, int kernel_num, int channels,
                                              int output_pooling_size,
                                              const int* output_idx,
                                              const float* output_features_grad,
                                              float* input_features_grad);

torch::Tensor voxel_sampling_feature_grad_gpu(
    torch::Tensor input_features,
    torch::Tensor output_idx,
    torch::Tensor output_features_grad
){
    CHECK_INPUT(input_features);
    CHECK_INPUT(output_idx);
    CHECK_INPUT(output_features_grad);

    float * input_features_ptr = input_features.data_ptr<float>();
    int * output_idx_ptr = output_idx.data_ptr<int>();
    float * output_features_grad_ptr = output_features_grad.data_ptr<float>();

    int input_point_num = input_features.size(0);
    int channels = input_features.size(1);
    int output_pooling_size = output_idx.size(2);
    int center_num = output_idx.size(0);
    int kernel_num = output_idx.size(1);

    torch::Tensor input_features_grad = torch::zeros(
        {input_point_num, channels}, 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(output_features_grad.device().type(), output_features_grad.device().index()).requires_grad(true));
    
    float * input_features_grad_ptr = input_features_grad.data_ptr<float>();

    voxel_sampling_feature_grad_gpu_launcher(center_num, kernel_num, channels,
                                                output_pooling_size,
                                                output_idx_ptr,
                                                output_features_grad_ptr,
                                                input_features_grad_ptr);
    return input_features_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxel_sampling_feature_gpu", &voxel_sampling_feature_gpu, "voxel_sampling_feature_gpu forward (CUDA)");
    m.def("voxel_sampling_feature_grad_gpu", &voxel_sampling_feature_grad_gpu, "voxel_sampling_feature_gpu backward (CUDA)");
}
