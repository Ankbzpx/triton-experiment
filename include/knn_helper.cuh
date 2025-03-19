// Heavily copied from https://github.com/tilmantroester/cudakdtree_jax_binding

#ifndef KNN_HELPER_CUH
#define KNN_HELPER_CUH

#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cukd/builder.h>
#include <cukd/knn.h>

const torch::Tensor QueryClosest(const torch::Tensor &input,
                                 const torch::Tensor &query);
#endif // KNN_HELPER_CUH