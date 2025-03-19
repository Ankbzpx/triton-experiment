#include "closest_query.cuh"

#define BATCH_SIZE 128

__global__ void ClosestPointKernel(int *d_indices, float3 *d_queries,
                                   int numQueries,
                                   const cukd::box_t<float3> *d_bounds,
                                   OrderedPoint3 *d_nodes, int numNodes,
                                   float cutOffRadius) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;

  using point_t = float3;
  point_t queryPos = d_queries[tid];
  cukd::FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int closestID = cukd::cct::fcp<OrderedPoint3, OrderedPoint3_traits>(
      queryPos, *d_bounds, d_nodes, numNodes, params);
  d_indices[tid] = d_nodes[closestID].idx;
}

const torch::Tensor QueryClosest(const torch::Tensor &input,
                                 const torch::Tensor &query) {
  int numInput = input.size(0);
  int numQueries = query.size(0);

  // We must copy because implicit tree will re-arange input data
  OrderedPoint3 *d_input;
  CUKD_CUDA_CHECK(
      cudaMallocManaged((void **)&d_input, numInput * sizeof(OrderedPoint3)));

  // **IMPORTANT** We cannot loop, as data is in device memory
  CopyKernel<<<cukd::divRoundUp(numInput, BATCH_SIZE), BATCH_SIZE>>>(
      d_input, reinterpret_cast<float3 *>(input.data_ptr<float>()), numInput);
  CUKD_CUDA_SYNC_CHECK();
  float3 *d_queries = reinterpret_cast<float3 *>(query.data_ptr<float>());

  cukd::box_t<float3> *d_bounds;
  CUKD_CUDA_CHECK(
      cudaMallocManaged((void **)&d_bounds, sizeof(cukd::box_t<float3>)));
  cukd::buildTree<OrderedPoint3, OrderedPoint3_traits>(d_input, numInput,
                                                       d_bounds);
  CUKD_CUDA_SYNC_CHECK();

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  torch::Tensor idxs = torch::zeros({numQueries}, opts);

  ClosestPointKernel<<<cukd::divRoundUp(numQueries, BATCH_SIZE), BATCH_SIZE>>>(
      idxs.data_ptr<int>(), d_queries, numQueries, d_bounds, d_input, numInput,
      100);
  CUKD_CUDA_SYNC_CHECK();

  return idxs;
}