#ifndef IMPORTANCE_SAMPLING_CUH
#define IMPORTANCE_SAMPLING_CUH

#include <c10/core/DeviceType.h>
#include <cstdint>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/random.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, uint32_t N_POS_DIMS, tcnn::HashType HASH_TYPE>
__global__ void
HashInputKernel(const uint32_t num_elements,
                const tcnn::GridOffsetTable offset_table,
                const uint32_t base_resolution, const T log2_per_level_scale,
                T *__restrict__ grid, tcnn::MatrixView<const T> positions_in) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_elements)
    return;

  const uint32_t level = blockIdx.y; // <- the level is the same for all threads

  grid += offset_table.data[level] * N_POS_DIMS;
  const uint32_t hashmap_size =
      offset_table.data[level + 1] - offset_table.data[level];

  const T scale =
      tcnn::grid_scale(level, log2_per_level_scale, base_resolution);
  const uint32_t resolution = tcnn::grid_resolution(scale);

  T pos[N_POS_DIMS];
  tcnn::uvec<N_POS_DIMS> pos_grid;

  TCNN_PRAGMA_UNROLL
  for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
    tcnn::pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,
                    tcnn::identity_fun);
  }

  const uint32_t hash_index = tcnn::grid_index<N_POS_DIMS, HASH_TYPE>(
      tcnn::GridType::Hash, hashmap_size, resolution, pos_grid);

  TCNN_PRAGMA_UNROLL
  for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
    grid[hash_index * N_POS_DIMS + dim] = pos_grid[dim];
  }
}

template <typename T, uint32_t N_POS_DIMS>
void CurdeNN(const torch::Tensor &input, const torch::Tensor &query) {
  uint32_t numInput = input.size(0);
  uint32_t numQueries = query.size(0);

  // TODO: Expose all configs
  uint32_t log2_hashmap_size = 19;
  uint32_t n_levels = 8;
  uint32_t base_resolution = 16;
  T per_level_scale = 2.0;

  tcnn::GridOffsetTable offset_table;

  uint32_t offset = 0;

  for (uint32_t i = 0; i < n_levels; ++i) {
    // Compute number of dense params required for the given level
    const uint32_t resolution = tcnn::grid_resolution(
        tcnn::grid_scale(i, std::log2(per_level_scale), base_resolution));

    uint32_t max_params = std::numeric_limits<uint32_t>::max() / 2;
    uint32_t params_in_level =
        std::pow((T)resolution, N_POS_DIMS) > (T)max_params
            ? max_params
            : tcnn::powi(resolution, N_POS_DIMS);

    // Make sure memory accesses will be aligned
    params_in_level = tcnn::next_multiple(params_in_level, 8u);

    // If hash table needs fewer params than dense, then use fewer and rely on
    // the hash.
    params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));

    offset_table.data[i] = offset;
    offset += params_in_level;
  }

  offset_table.data[n_levels] = offset;
  offset_table.size = n_levels + 1;

  uint32_t n_params = offset_table.data[n_levels] * N_POS_DIMS;

  // Initialize the hashgrid from the GPU, because the number of parameters can
  // be quite large.
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor params = INFINITY * torch::ones({n_params}, opts);

  // Idea: each block only takes care of _one_ hash level (but may iterate over
  // multiple input elements). This way, only one level of the hashmap needs to
  // fit into caches at a time (and it reused for consecutive elements) until it
  // is time to process the next level.

  static constexpr uint32_t N_THREADS_HASHGRID = 512;
  const dim3 blocks_hashgrid = {
      tcnn::div_round_up(numInput, N_THREADS_HASHGRID), n_levels, 1};

  tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor> input_matrix(
      input.data_ptr<T>(), N_POS_DIMS, numInput);

  HashInputKernel<T, N_POS_DIMS, tcnn::HashType::CoherentPrime>
      <<<blocks_hashgrid, N_THREADS_HASHGRID>>>(
          numInput, offset_table, base_resolution, std::log2(per_level_scale),
          params.data_ptr<T>(), input_matrix.view());

  std::cout << (params != INFINITY).sum() << std::endl;
}
#endif // IMPORTANCE_SAMPLING_CUH