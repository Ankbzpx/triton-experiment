#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <torch/extension.h>

#include "closest_query.cuh"
#include "importance_sampling.cuh"

int main(int argc, char *argv[]) {
  torch::manual_seed(0);

  torch::Tensor xyz1 = torch::randn({17, 3}, torch::kCUDA);
  torch::Tensor xyz2 = torch::randn({33, 3}, torch::kCUDA);

  const at::cuda::CUDAGuard device_guard{xyz1.device()};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  std::cout << CurdeNN(stream, xyz2, xyz1) << std::endl;
  std::cout << QueryClosest(stream, xyz2.contiguous(), xyz1.contiguous()) << std::endl;

  return 0;
}