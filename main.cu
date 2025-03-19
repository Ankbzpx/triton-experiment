#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#include "closest_query.cuh"
#include <tiny-cuda-nn/encodings/grid.h>

int main(int argc, char *argv[]) {
  torch::manual_seed(0);
  
  torch::Tensor xyz1 = torch::randn({17, 3}, torch::kCUDA);
  torch::Tensor xyz2 = torch::randn({33, 3}, torch::kCUDA);

  std::cout << QueryClosest(xyz2.contiguous(), xyz1.contiguous()) << std::endl;

  return 0;
}