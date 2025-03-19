#include "cukd/cukd-math.h"
#include "knn_helper.cuh"
#include <torch/types.h>

__global__ void d_fcp(int *d_indices, float3 *d_queries, int numQueries,
                      /*! the world bounding box computed by the builder */
                      const cukd::box_t<float3> *d_bounds, float3 *d_nodes,
                      int numNodes, float cutOffRadius) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;

  using point_t = float3;
  point_t queryPos = d_queries[tid];
  cukd::FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int closestID =
      cukd::cct::fcp(queryPos, *d_bounds, d_nodes, numNodes, params);
  d_indices[tid] = closestID;
}

const torch::Tensor QueryClosest(const torch::Tensor &input,
                                 const torch::Tensor &query) {
  int numInput = input.size(0);
  int numQueries = query.size(0);

  float3 *d_input = reinterpret_cast<float3 *>(input.data_ptr<float>());
  float3 *d_queries = reinterpret_cast<float3 *>(query.data_ptr<float>());

  cukd::box_t<float3> *d_bounds;
  cudaMallocManaged((void **)&d_bounds, sizeof(cukd::box_t<float3>));
  cukd::buildTree(d_input, numInput, d_bounds);
  CUKD_CUDA_SYNC_CHECK();

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  torch::Tensor idxs = torch::zeros({numQueries}, opts);

  int bs = 128;
  int nb = cukd::divRoundUp((int)numQueries, bs);
  d_fcp<<<nb, bs>>>(idxs.data_ptr<int>(), d_queries, numQueries, d_bounds,
                    d_input, numInput, 100);
  CUKD_CUDA_SYNC_CHECK();

  return idxs;
}