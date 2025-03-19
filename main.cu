#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#include "knn_helper.cuh"
#include <tiny-cuda-nn/encodings/grid.h>

int main(int argc, char *argv[]) {
  torch::manual_seed(0);
  torch::Tensor input = torch::randn({17, 3}, torch::kCUDA);
  torch::Tensor query = torch::randn({33, 3}, torch::kCUDA);
  
  std::cout << QueryClosest(input, query) << std::endl;

  std::cout << input << std::endl;
  std::cout << query << std::endl;

  return 0;
}