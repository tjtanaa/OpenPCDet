/* Unique Operation
 * GPU implementation of unique operation.
 * Created by Zhaoyu SU
 * Ported by Tun Jian TAN
 * All Rights Reserved. Nov., 2020.
 */
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
// #include <iostream>
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

void grid_sampling_gpu_launcher(int batch_size, int input_point_num,
                                int grid_w, int grid_l, int grid_h,
                                float res_x, float res_y, float res_z,
                                const float* input_coors,
                                const int* input_num_list,
                                int* input_accu_list,
                                int* output_idx_temp,
                                int* output_num_list,
                                int* grid_buffer);



std::vector<torch::Tensor> grid_sampling_gpu(
    // torch::Tensor dimension, 
    int grid_w, int grid_l, int grid_h,
    float res_x, float res_y, float res_z,
    torch::Tensor input_coors, torch::Tensor input_num_list 
    // torch::Tensor output_idx, torch::Tensor output_num_list
    ){
    // params input_coors: (N, 3) [x, y, z]
    // params input_num_list: (B) [number of points in each batch] 
    // params output_idx: (N) [index of points to be kept]
    // params output_num_list: (B) [number of points in each batch] 
    // params dimension: (3) [Dx, Dy, Dz]
    // params resolution: (3) [res_x, res_y, res_z]

    // CHECK_INPUT(dimension);
    // CHECK_INPUT(resolution);
    CHECK_INPUT(input_coors);
    CHECK_INPUT(input_num_list);
    // CHECK_INPUT(output_idx);
    // CHECK_INPUT(output_num_list);

    // assert(("grid_sampling_gpu ERROR: resolution > 0\n", resolution > 0 ));

    // const float * dimension_ptr = dimension.data_ptr<float>();
    const float * input_coors_ptr = input_coors.data_ptr<float>();
    const int * input_num_list_ptr = input_num_list.data_ptr<int>();

    torch::Tensor output_num_list = torch::zeros_like(input_num_list);
    int * output_num_list_ptr = output_num_list.data_ptr<int>();

    int input_point_num = input_coors.size(0);
    int batch_size = input_num_list.size(0);
    // int grid_w = int(ceil(dimension_ptr[0] / resolution));
    // int grid_l = int(ceil(dimension_ptr[1] / resolution));
    // int grid_h = int(ceil(dimension_ptr[2] / resolution));

    // if (INT_MAX / grid_h / grid_l / grid_w < batch_size){
    //     fprintf("grid_sampling_gpu ERROR: size of grid buffer %d x [%d x %d x %d] exceeds INT32 range: %d.\n",
    //             batch_size, grid_w, grid_l, grid_h, INT_MAX);}
    // std::cout << std::to_string(grid_w) >> std::endl;

    int* input_num_list_ptr_host = (int*)malloc(batch_size*sizeof(int));
    cudaMemcpy(input_num_list_ptr_host, input_num_list_ptr, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    int* input_accu_list_ptr_host = (int*)malloc(batch_size  *sizeof(int));
    input_accu_list_ptr_host[0] = 0;
    for (int i=1; i<batch_size; i++)
        input_accu_list_ptr_host[i] = input_accu_list_ptr_host[i-1] + input_num_list_ptr_host[i-1];
    
    
    torch::Tensor input_accu_list = torch::zeros({batch_size}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * input_accu_list_ptr = input_accu_list.data_ptr<int>();
    // int *input_accu_list_ptr = nullptr;
    // cudaMalloc((void**)&input_accu_list_ptr, batch_size * sizeof(int));
    cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_size * sizeof(int), cudaMemcpyHostToDevice);


    torch::Tensor output_idx_temp = torch::zeros({input_point_num}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * output_idx_temp_ptr = output_idx_temp.data_ptr<int>();
    // int *output_idx_temp_ptr;
    // cudaMalloc((void**)&output_idx_temp_ptr, input_point_num * sizeof(int));
    // cudaMemset(output_idx_temp_ptr, 0, input_point_num * sizeof(int));

    
    torch::Tensor grid_buffer = torch::zeros({batch_size * grid_w * grid_l * grid_h}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    int * grid_buffer_ptr = grid_buffer.data_ptr<int>();
    // int *grid_buffer_ptr;
    // cudaMalloc((void**)&grid_buffer_ptr, batch_size * grid_w * grid_l * grid_h * sizeof(int));
    // cudaMemset(grid_buffer_ptr, 0, batch_size * grid_w * grid_l * grid_h * sizeof(int));


    grid_sampling_gpu_launcher(batch_size, input_point_num,
                                grid_w, grid_l, grid_h,
                                res_x, res_y, res_z,
                                input_coors_ptr,
                                input_num_list_ptr,
                                input_accu_list_ptr,
                                output_idx_temp_ptr,
                                output_num_list_ptr,
                                grid_buffer_ptr);


    int* output_idx_temp_ptr_host = (int*)malloc(input_point_num*sizeof(int));
    cudaMemcpy(output_idx_temp_ptr_host, output_idx_temp_ptr, input_point_num*sizeof(int), cudaMemcpyDeviceToHost);
    int* output_num_list_ptr_host = (int*)malloc(batch_size*sizeof(int));
    cudaMemcpy(output_num_list_ptr_host, output_num_list_ptr, batch_size*sizeof(int), cudaMemcpyDeviceToHost);


    int output_count = 0;
    for (int i=0; i<batch_size; i++) {
        output_count += output_num_list_ptr_host[i];
    }
//        printf("******************input shape = %d************************\n", input_point_num);
//        printf("******************output shape = %d************************\n", output_count);
    int target_count = 0;
    int source_count = 0;
    torch::Tensor output_idx = torch::zeros({output_count}, 
                        torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(input_num_list.device().type(), input_num_list.device().index()).requires_grad(false));
    
    int * output_idx_ptr = output_idx.data_ptr<int>();
    for (int i=0; i<batch_size; i++) {
        int cpy_size = output_num_list_ptr_host[i] * sizeof(int);
        cudaMemcpy(&output_idx_ptr[target_count], &output_idx_temp_ptr_host[source_count], cpy_size, cudaMemcpyHostToDevice);
//            for (int j=0; j<output_num_list_ptr_host[i]; j++) {
//                if (output_idx_temp_ptr_host[source_count + j] < source_count || output_idx_temp_ptr_host[source_count + j] >= source_count + input_num_list_ptr_host[i])
//                printf("Batch-%d point Id %d exceed range [%d, %d]\n", i, output_idx_temp_ptr_host[source_count + j], source_count, source_count + input_num_list_ptr_host[i]);
//                printf("Batch-%d point Id[%d] = %d.\n", i, j, output_idx_temp_ptr_host[source_count + j]);
//            }
        target_count += output_num_list_ptr_host[i];
        source_count += input_num_list_ptr_host[i];

    }
    free(output_idx_temp_ptr_host);
    free(output_num_list_ptr_host);
    free(input_accu_list_ptr_host);
    
    // cudaFree(input_accu_list_ptr);
    // cudaFree(output_idx_temp_ptr);
    // cudaFree(grid_buffer_ptr);

    return {output_idx, output_num_list};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_sampling_gpu", &grid_sampling_gpu, "grid_sampling_gpu forward (CUDA)");
}

