// Heavily copied from https://github.com/tilmantroester/cudakdtree_jax_binding

#ifndef KNN_HELPER_CUH
#define KNN_HELPER_CUH

#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cukd/builder.h>
#include <cukd/fcp.h>

template <typename point_t> struct OrderedPoint {
  point_t position;
  int idx;
};

template <typename point_t>
struct OrderedPoint_traits : public cukd::default_data_traits<point_t> {
  using data_t = OrderedPoint<point_t>;
  using point_traits = cukd::point_traits<point_t>;
  using scalar_t = typename point_traits::scalar_t;

  static inline __device__ __host__ const point_t &
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

using OrderedPoint3 = OrderedPoint<float3>;
using OrderedPoint3_traits = OrderedPoint_traits<float3>;

const torch::Tensor QueryClosest(const torch::Tensor &input,
                                 const torch::Tensor &query);
#endif // KNN_HELPER_CUH