#include "cublas_v2.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


torch::Tensor run_fnn_forward(const torch::Tensor& input, const torch::Tensor& weight1, const torch::Tensor& weight2, const torch::optional<torch::Tensor> bias1, const torch::optional<torch::Tensor> bias2) {

    // std::cout << __FILE__ << " " << __FUNCTION__ << std::endl;
    
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashFFN only supports Ampere GPUs or newer.");

    auto input_sizes = input.sizes();
    const int input_dim = input_sizes[0];
    const int input_features = input_sizes[1];

    auto weight1_sizes = weight1.sizes();
    const int weight1_in_features = weight1_sizes[1];
    const int weight1_out_features = weight1_sizes[0];

    auto weight2_sizes = weight2.sizes();
    const int weight2_in_features = weight2_sizes[1];
    const int weight2_out_features = weight2_sizes[0];
    
    {
        auto input_dtype = input.dtype();
        TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kBFloat16, "FlashFFN only support fp16 and bf16 data type");
        TORCH_CHECK(input.is_cuda(), "input tensor must be on CUDA device");
        TORCH_CHECK(input.stride(-1) == 1, "input tensor must have contiguous last dimension");

        if (input_dtype == torch::kBFloat16) {
            TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
        }
        TORCH_CHECK(weight1.dtype() == input_dtype, "weight1 tensor and input tensor must have the same dtype");
        TORCH_CHECK(weight1.is_cuda(), "weight1 tensor must be on CUDA device");
        TORCH_CHECK(weight1.stride(-1) == 1, "weight1 tensor must have contiguous last dimension");

        TORCH_CHECK(weight2.dtype() == input_dtype, "weight2 tensor and input tensor must have the same dtype");
        TORCH_CHECK(weight2.is_cuda(), "weight2 tensor must be on CUDA device");
        TORCH_CHECK(weight2.stride(-1) == 1, "weight2 tensor must have contiguous last dimension");

        TORCH_CHECK(bias1.has_value() == bias2.has_value(), "bias1 and bias2 must both be present or both be absent");

        if (bias1.has_value() && bias2.has_value()) {
            auto bias1_value = bias1.value();
            auto bias2_value = bias2.value();
            CHECK_SHAPE(bias1_value, weight1_out_features);
            CHECK_SHAPE(bias2_value, weight2_out_features);
            TORCH_CHECK(bias1_value.dtype() == input_dtype, "bias1 tensor and input tensor must have the same dtype");
            TORCH_CHECK(bias1_value.is_cuda(), "bias1 tensor must be on CUDA device");
            TORCH_CHECK(bias1_value.stride(-1) == 1, "bias1 tensor must have contiguous last dimension");

            TORCH_CHECK(bias2_value.dtype() == input_dtype, "bias2 tensor and input tensor must have the same dtype");
            TORCH_CHECK(bias2_value.is_cuda(), "bias2 tensor must be on CUDA device");
            TORCH_CHECK(bias2_value.stride(-1) == 1, "bias2 tensor must have contiguous last dimension");
        }

        TORCH_CHECK(input_dim > 0, "input accumulated seq len must be greater than 0");
        TORCH_CHECK(weight1_in_features == input_features, "weight1_in_features must match input_features");
        TORCH_CHECK(weight2_in_features == weight1_out_features, "weight2_in_features must match weight1_out_features");
        TORCH_CHECK(weight2_out_features == input_features, "weight2_out_features must match input_features");
    }

    torch::Tensor output = torch::empty_like(input);
    torch::Tensor linear1_output = torch::empty(std::vector<int64_t>({input_dim, weight1_out_features}), input.options());

    // printf("output size in bytes: %d \n", output.element_size() * output.numel());
    // printf("linear1_output size in bytes: %d \n", linear1_output.element_size() * linear1_output.numel());
    

    at::cuda::CUDAGuard device_guard{(char)input.get_device()};

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const __half alpha = 1.;
    const __half beta = 0.;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t stat;

    {
        const int M = input_dim;
        const int N = weight1_out_features;
        const int K = input_features;
        // Row_M: [M, K] x [N, K]^T = [M, N] 
        stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, (const void *)&alpha,
                    (const void *)weight1.data_ptr(), CUDA_R_16F, K, (const void *)input.data_ptr(), CUDA_R_16F, K,
                    (const void *)&beta, linear1_output.data_ptr(), CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
    }

    {
        const int M = input_dim;
        const int N = weight2_out_features;
        const int K = weight2_in_features;
        // Row_M: [M, N] x [K, N]^T = [M, K] 
        stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, (const void *)&alpha,
                    (const void *)weight2.data_ptr(), CUDA_R_16F, K, (const void *)linear1_output.data_ptr(), CUDA_R_16F, K,
                    (const void *)&beta, output.data_ptr(), CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
    }

    cublasDestroy(handle);

    return output;
}

TORCH_LIBRARY(cuffn, m) {
//   m.doc() = "cuffn: an attempt for the fastest FFN implementation";
  m.def("run_fnn_forward", &run_fnn_forward);
}