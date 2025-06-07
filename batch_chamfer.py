import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.distance import cdist
from typing import List

import triton
import triton.language as tl

from icecream import ic


@triton.jit
def nm_dist_kernel(xyz1_ptr, xyz2_ptr, lock_ptr, dists_ptr, indices_ptr, B, N,
                   M, xyz1_stride_b, xyz1_stride_n, xyz1_stride_d,
                   xyz2_stride_b, xyz2_stride_m, xyz2_stride_d, dist_stride_b,
                   dist_stride_n, indices_stride_b, indices_stride_n,
                   lock_stride_b, lock_stride_n, BLOCK_SIZE_B: tl.constexpr,
                   BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                   GROUP_SIZE: tl.constexpr):

    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    # Let's do it later
    # More L2 cache friendly launch
    num_pid_n = tl.num_programs(axis=1)
    num_pid_m = tl.num_programs(axis=2)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE)

    base_b = pid_b * BLOCK_SIZE_B
    base_n = pid_n * BLOCK_SIZE_N
    base_m = pid_m * BLOCK_SIZE_M

    batch_base_b = base_b + tl.arange(0, BLOCK_SIZE_B)
    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)

    batch_n_mask = (batch_base_b[:, None] < B) & (batch_base_n[None, :] < N)

    xyz1_x = tl.load(xyz1_ptr + batch_base_b[:, None] * xyz1_stride_b +
                     batch_base_n[None, :] * xyz1_stride_n,
                     mask=batch_n_mask,
                     other=100)
    xyz1_y = tl.load(xyz1_ptr + batch_base_b[:, None] * xyz1_stride_b +
                     batch_base_n[None, :] * xyz1_stride_n + xyz1_stride_d,
                     mask=batch_n_mask,
                     other=100)
    xyz1_z = tl.load(xyz1_ptr + batch_base_b[:, None] * xyz1_stride_b +
                     batch_base_n[None, :] * xyz1_stride_n + 2 * xyz1_stride_d,
                     mask=batch_n_mask,
                     other=100)

    batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
    batch_m_mask = (batch_base_b[:, None] < B) & (batch_base_m[None, :] < M)
    xyz2_x = tl.load(xyz2_ptr + batch_base_b[:, None] * xyz2_stride_b +
                     batch_base_m[None, :] * xyz2_stride_m,
                     mask=batch_m_mask,
                     other=-100)
    xyz2_y = tl.load(xyz2_ptr + batch_base_b[:, None] * xyz2_stride_b +
                     batch_base_m[None, :] * xyz2_stride_m + xyz2_stride_d,
                     mask=batch_m_mask,
                     other=-100)
    xyz2_z = tl.load(xyz2_ptr + batch_base_b[:, None] * xyz2_stride_b +
                     batch_base_m[None, :] * xyz2_stride_m + 2 * xyz2_stride_d,
                     mask=batch_m_mask,
                     other=-100)

    x2 = xyz1_x[:, :, None] - xyz2_x[:, None, :]
    y2 = xyz1_y[:, :, None] - xyz2_y[:, None, :]
    z2 = xyz1_z[:, :, None] - xyz2_z[:, None, :]
    d = x2 * x2 + y2 * y2 + z2 * z2

    best_d = tl.min(d, axis=2)

    # ​TODO: sqrt depends on SLU. Let pytorch handle it for now
    # best_d = tl.sqrt(best_d)
    best_idx = tl.argmin(d, axis=2) + base_m

    lock = lock_ptr + pid_b * lock_stride_b + pid_n * lock_stride_n
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass

    cur_best_d = tl.load(dists_ptr + batch_base_b[:, None] * dist_stride_b +
                         batch_base_n[None, :] * dist_stride_n,
                         mask=batch_n_mask)

    # Handle zero initialization in JAX
    # FIXME: The safer option is to use another lock for first occuring pid_n initialization
    out_mask = ((best_d < cur_best_d) | (cur_best_d == 0)) & batch_n_mask
    tl.store(dists_ptr + batch_base_b[:, None] * dist_stride_b +
             batch_base_n[None, :] * dist_stride_n,
             best_d,
             mask=out_mask)
    tl.store(indices_ptr + batch_base_b[:, None] * indices_stride_b +
             batch_base_n[None, :] * indices_stride_n,
             best_idx,
             mask=out_mask)

    # Release lock
    tl.atomic_xchg(lock, 0)


def nm_dist(xyz1: torch.Tensor, xyz2: torch.Tensor):
    assert xyz1.shape[-1] == xyz2.shape[-1], "Incompatible dimensions"
    assert xyz1.is_contiguous(), "Matrix xyz1 must be contiguous"
    assert xyz2.is_contiguous(), "Matrix xyz2 must be contiguous"

    B, N, D = xyz1.shape
    B, M, D = xyz2.shape

    dists = torch.zeros((B, N), device=xyz1.device, dtype=xyz1.dtype)
    indices = torch.zeros((B, N), device=xyz1.device, dtype=torch.int32)

    grid = lambda META: (triton.cdiv(B, META['BLOCK_SIZE_B']),
                         triton.cdiv(N, META['BLOCK_SIZE_N']),
                         triton.cdiv(M, META['BLOCK_SIZE_M']))

    configs = {
        'BLOCK_SIZE_B': 16,
        'BLOCK_SIZE_N': 16,
        'BLOCK_SIZE_M': 512,
        'GROUP_SIZE': 16,
        "num_warps": 2,
        "num_stages": 3
    }

    lock = torch.zeros((triton.cdiv(
        B, configs['BLOCK_SIZE_B']), triton.cdiv(N, configs['BLOCK_SIZE_N'])),
                       device=xyz1.device,
                       dtype=torch.int32)

    nm_dist_kernel[grid](xyz1, xyz2, lock, dists, indices, B, N, M,
                         xyz1.stride(0), xyz1.stride(1), xyz1.stride(2),
                         xyz2.stride(0), xyz2.stride(1), xyz2.stride(2),
                         dists.stride(0), dists.stride(1), indices.stride(0),
                         indices.stride(1), lock.stride(0), lock.stride(1),
                         **configs)
    return dists, indices


def chamfer_distance(xyz1: torch.Tensor, xyz2: torch.Tensor):
    dist1, idx1 = nm_dist(xyz1, xyz2)
    dist2, idx2 = nm_dist(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


if __name__ == "__main__":
    import mash_cpp

    torch.manual_seed(0)

    xyz1 = torch.randn(2, 17, 3).cuda()
    xyz2 = torch.randn(2, 15, 3).cuda()

    dist1_mashcpp, dist2_mashcpp, idx1_mashcpp, idx2_mashcpp = mash_cpp.toChamferDistance(
        xyz1, xyz2)

    dist1, idx1 = nm_dist(xyz1, xyz2)
    dist2, idx2 = nm_dist(xyz2, xyz1)

    ic(torch.allclose(dist1, dist1_mashcpp))
    ic(torch.allclose(idx1, idx1_mashcpp))
    ic(torch.allclose(dist2, dist2_mashcpp))
    ic(torch.allclose(idx2, idx2_mashcpp))

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["B", "N", "M"],
            x_vals=[np.power(10, i) for i in range(1, 3)],
            line_arg="provider",
            line_vals=["Triton", "MASH"],
            line_names=["Triton", "MASH"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="NMDist Performance",
            args={}))

    @triton.testing.perf_report(configs)
    def benchmark(B, M, N, provider):
        xyz1 = torch.randn(10 * B, M, 3).cuda()
        xyz2 = torch.randn(10 * B, N, 3).cuda()
        quantiles = [0.5, 0.2, 0.8]
        if provider == "MASH":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mash_cpp.toChamferDistance(xyz1, xyz2), quantiles=quantiles)
        if provider == "Triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: chamfer_distance(xyz1, xyz2), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)
