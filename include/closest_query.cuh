// Heavily copied from https://github.com/tilmantroester/cudakdtree_jax_binding

#ifndef KNN_HELPER_CUH
#define KNN_HELPER_CUH

#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cukd/builder.h>
#include <cukd/fcp.h>

template <typename PointT> struct OrderedPoint {
  PointT position;
  int idx;
};

template <typename PointT>
struct OrderedPoint_traits : public cukd::default_data_traits<PointT> {
  using data_t = OrderedPoint<PointT>;
  using point_traits = cukd::point_traits<PointT>;
  using scalar_t = typename point_traits::scalar_t;

  static inline __device__ __host__ const PointT &
  get_point(const data_t &data) {
    return data.position;
  }

  static inline __device__ __host__ scalar_t get_coord(const data_t &data,
                                                       int dim) {
    return cukd::get_coord(get_point(data), dim);
  }

  enum { has_explicit_dim = false };
  static inline __device__ int get_dim(const data_t &) { return -1; }
};

template <typename PointT>
__global__ void CopyKernel(OrderedPoint<PointT> *points, PointT *positions,
                           int n_points) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_points)
    return;
  points[tid].position = positions[tid];
  points[tid].idx = tid;
}

template <typename PointT>
__global__ void
ClosestPointKernel(int *d_indices, PointT *d_queries, int numQueries,
                   const cukd::box_t<PointT> *d_bounds,
                   OrderedPoint<PointT> *d_nodes, int numNodes) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numQueries)
    return;
  PointT queryPos = d_queries[tid];
  cukd::FcpSearchParams params;
  int closestID =
      cukd::cct::fcp<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
          queryPos, *d_bounds, d_nodes, numNodes, params);
  d_indices[tid] = d_nodes[closestID].idx;
}

template <typename T = float, typename PointT = float3,
          uint32_t BATCH_SIZE = 128>
const torch::Tensor QueryClosest(cudaStream_t stream,
                                 const torch::Tensor &input,
                                 const torch::Tensor &query) {
  uint32_t numInput = input.size(0);
  uint32_t numQueries = query.size(0);

  // We must copy because implicit tree will re-arange input data
  OrderedPoint<PointT> *d_input;
  CUKD_CUDA_CHECK(cudaMallocAsync(
      (void **)&d_input, numInput * sizeof(OrderedPoint<PointT>), stream));

  // **IMPORTANT** We cannot loop, as data is in device memory
  CopyKernel<<<cukd::divRoundUp(numInput, BATCH_SIZE), BATCH_SIZE, 0, stream>>>(
      d_input, reinterpret_cast<PointT *>(input.data_ptr<T>()), numInput);
  CUKD_CUDA_SYNC_CHECK();
  PointT *d_queries = reinterpret_cast<PointT *>(query.data_ptr<T>());

  cukd::box_t<PointT> *d_bounds;
  CUKD_CUDA_CHECK(
      cudaMallocAsync((void **)&d_bounds, sizeof(cukd::box_t<PointT>), stream));
  cukd::buildTree<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
      d_input, numInput, d_bounds, stream);
  CUKD_CUDA_SYNC_CHECK();

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  torch::Tensor idxs = torch::zeros({numQueries}, opts);

  ClosestPointKernel<<<cukd::divRoundUp(numQueries, BATCH_SIZE), BATCH_SIZE, 0,
                       stream>>>(idxs.data_ptr<int>(), d_queries, numQueries,
                                 d_bounds, d_input, numInput);
  CUKD_CUDA_SYNC_CHECK();
  return idxs;
}

#endif // KNN_HELPER_CUH