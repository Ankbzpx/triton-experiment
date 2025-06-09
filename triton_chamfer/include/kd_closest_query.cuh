// Heavily copied from https://github.com/tilmantroester/cudakdtree_jax_binding

#ifndef KD_CLOSEST_QUERY_CUH
#define KD_CLOSEST_QUERY_CUH

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
                           int n_batches, int n_points) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int nid = threadIdx.y + blockIdx.y * blockDim.y;

  if ((bid >= n_batches) || (nid >= n_points))
    return;

  // Row major
  int pid = bid * n_points + nid;
  points[pid].position = positions[pid];
  // Batch local index
  points[pid].idx = nid;
}

template <typename T, typename PointT>
__global__ void
ClosestPointKernel(T *d_dists, int *d_indices, PointT *d_queries, int n_batches,
                   int n_queries, const cukd::box_t<PointT> *d_bounds,
                   OrderedPoint<PointT> *d_nodes, int n_points) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int mid = threadIdx.y + blockIdx.y * blockDim.y;

  if ((bid >= n_batches) || (mid >= n_queries))
    return;

  // Row major
  int qid = bid * n_queries + mid;
  PointT queryPos = d_queries[qid];
  cukd::FcpSearchParams params;
  // Local closest index
  int closestID =
      cukd::cct::fcp<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
          queryPos, *(d_bounds + bid), d_nodes + bid * n_points, n_points,
          params);
  int pid = bid * n_points + closestID;
  int idx = d_nodes[pid].idx;
  PointT inputPos = d_nodes[pid].position;
  d_dists[qid] = std::pow(queryPos.x - inputPos.x, 2) +
                 std::pow(queryPos.y - inputPos.y, 2) +
                 std::pow(queryPos.z - inputPos.z, 2);
  d_indices[qid] = idx;
}

template <typename T = float, typename PointT = float3,
          uint32_t BATCH_SIZE_B = 32, uint32_t BATCH_SIZE_N = 16,
          uint32_t BATCH_SIZE_M = 16>
std::vector<torch::Tensor> KDQueryClosest(cudaStream_t stream,
                                          const torch::Tensor &input,
                                          const torch::Tensor &query) {
  uint32_t numBatches = input.size(0);
  uint32_t numInput = input.size(1);
  uint32_t numQueries = query.size(1);

  // We must copy because implicit tree will re-arange input data
  OrderedPoint<PointT> *d_input;
  CUKD_CUDA_CHECK(cudaMallocAsync(
      (void **)&d_input, numBatches * numInput * sizeof(OrderedPoint<PointT>),
      stream));

  // **IMPORTANT** We cannot loop, as data is in device memory
  CopyKernel<<<dim3(cukd::divRoundUp(numBatches, BATCH_SIZE_B),
                    cukd::divRoundUp(numInput, BATCH_SIZE_N)),
               dim3(BATCH_SIZE_B, BATCH_SIZE_N), 0, stream>>>(
      d_input, reinterpret_cast<PointT *>(input.data_ptr<T>()), numBatches,
      numInput);
  CUKD_CUDA_SYNC_CHECK();
  PointT *d_queries = reinterpret_cast<PointT *>(query.data_ptr<T>());

  cukd::box_t<PointT> *d_bounds;
  CUKD_CUDA_CHECK(cudaMallocAsync(
      (void **)&d_bounds, numBatches * sizeof(cukd::box_t<PointT>), stream));

  for (int bid = 0; bid < numBatches; bid++) {
    cukd::buildTree<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
        d_input + bid * numInput, numInput, d_bounds + bid, stream);
  }
  CUKD_CUDA_SYNC_CHECK();

  const torch::TensorOptions distOpts =
      torch::TensorOptions().dtype(query.dtype()).device(query.device());
  torch::Tensor dists = torch::zeros({numBatches, numQueries}, distOpts);

  const torch::TensorOptions idxOpts =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  torch::Tensor idxs = torch::zeros({numBatches, numQueries}, idxOpts);

  ClosestPointKernel<<<dim3(cukd::divRoundUp(numBatches, BATCH_SIZE_B),
                            cukd::divRoundUp(numQueries, BATCH_SIZE_M)),
                       dim3(BATCH_SIZE_B, BATCH_SIZE_M), 0, stream>>>(
      dists.data_ptr<T>(), idxs.data_ptr<int>(), d_queries, numBatches,
      numQueries, d_bounds, d_input, numInput);
  CUKD_CUDA_SYNC_CHECK();
  return {dists, idxs};
}

#endif // KD_CLOSEST_QUERY_CUH