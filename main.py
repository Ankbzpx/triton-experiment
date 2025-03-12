import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.distance import cdist

import mash_cpp
import triton
import triton.language as tl

from icecream import ic


def closest_neighbour_sp(ref_pts, query_pts):
    dist_mat = cdist(ref_pts, query_pts)
    ref_closest_index_sp = np.argmin(dist_mat, axis=1)
    ref_closest_dist_sp = np.min(dist_mat, axis=1)
    query_closest_index_sp = np.argmin(dist_mat, axis=0)
    query_closest_dist_sp = np.min(dist_mat, axis=0)

    return ref_closest_dist_sp, ref_closest_index_sp, query_closest_dist_sp, query_closest_index_sp


@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_A_m, stride_A_k,
                  stride_B_k, stride_B_n, stride_C_m, stride_C_n,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # (BLOCK_SIZE_M, )
    offset_A_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # (BLOCK_SIZE_N, )
    offset_B_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # We later use mask to filter out
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # First BLOCK_SIZE_K address, A[m:m+BLOCK_SIZE_M, 0:BLOCK_SIZE_K]
    # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    A_ptrs = A_ptr + (offset_A_m[:, None] * stride_A_m +
                      offs_k[None, :] * stride_A_k)
    # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_B_k +
                      offset_B_n[None, :] * stride_B_n)

    C_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        A_block = tl.load(A_ptrs,
                          mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K),
                          other=0.0)
        B_block = tl.load(B_ptrs,
                          mask=offs_k[:, None] < (K - k * BLOCK_SIZE_K),
                          other=0.0)
        # **VERY IMPORTANT** Triton defaults to tf32 for device that has tensor cores
        C_block = tl.dot(A_block, B_block, C_block, allow_tf32=False)
        # Offset index by BLOCK_SIZE_K, A[m:m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]
        #   i.e. offset entry by BLOCK_SIZE_K * stride_A_k
        A_ptrs += BLOCK_SIZE_K * stride_A_k
        B_ptrs += BLOCK_SIZE_K * stride_B_k

    offset_C_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_C_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C_ptr + stride_C_m * offset_C_m[:,
                                             None] + stride_C_n * offset_C_n[
                                                 None, :]
    C_mask = (offset_C_m[:, None] < M) & (offset_C_n[None, :] < N)
    tl.store(C_ptrs, C_block, mask=C_mask)


def matmul(A: torch.Tensor, B: torch.Tensor):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                         triton.cdiv(N, META['BLOCK_SIZE_N']))

    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}

    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0),
                        A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                        C.stride(1), **config)
    return C


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 32
            },
            num_warps=2,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 64
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 64
            },
            num_warps=2,
            num_stages=3),
    ]


@triton.autotune(
    # configs=get_cuda_autotune_config(),
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 512,
                'GROUP_SIZE': 16
            },
            num_warps=2,
            num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def nm_dist_kernel(xyz1_ptr, xyz2_ptr, dist_ptr, indices_ptr, lock_ptr, N, M,
                   xyz1_stride_n, xyz1_stride_d, xyz2_stride_m, xyz2_stride_d,
                   dist_stride_n, indices_stride_n, lock_stride_n,
                   BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                   GROUP_SIZE: tl.constexpr):

    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # More L2 cache friendly launch
    num_pid_n = tl.num_programs(axis=0)
    num_pid_m = tl.num_programs(axis=1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE)

    base_n = pid_n * BLOCK_SIZE_N
    base_m = pid_m * BLOCK_SIZE_M

    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
    batch_n_mask = batch_base_n < N
    xyz1_x = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n,
                     mask=batch_n_mask,
                     other=100)
    xyz1_y = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n + xyz1_stride_d,
                     mask=batch_n_mask,
                     other=100)
    xyz1_z = tl.load(xyz1_ptr + batch_base_n * xyz1_stride_n +
                     2 * xyz1_stride_d,
                     mask=batch_n_mask,
                     other=100)

    batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
    batch_m_mask = batch_base_m < M
    xyz2_x = tl.load(xyz2_ptr + batch_base_m * xyz2_stride_m,
                     mask=batch_m_mask,
                     other=-100)
    xyz2_y = tl.load(xyz2_ptr + batch_base_m * xyz2_stride_m + xyz2_stride_d,
                     mask=batch_m_mask,
                     other=-100)
    xyz2_z = tl.load(xyz2_ptr + batch_base_m * xyz2_stride_m +
                     2 * xyz2_stride_d,
                     mask=batch_m_mask,
                     other=-100)

    x2 = xyz1_x[:, None] - xyz2_x[None, :]
    y2 = xyz1_y[:, None] - xyz2_y[None, :]
    z2 = xyz1_z[:, None] - xyz2_z[None, :]
    d = x2 * x2 + y2 * y2 + z2 * z2

    best_d = tl.sqrt(tl.min(d, axis=1))
    best_idx = tl.argmin(d, axis=1) + base_m

    lock = lock_ptr + pid_n * lock_stride_n
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass

    cur_best_d = tl.load(dist_ptr + batch_base_n * dist_stride_n,
                         mask=batch_n_mask)
    out_mask = (best_d < cur_best_d) & batch_n_mask
    tl.store(dist_ptr + batch_base_n * dist_stride_n, best_d, mask=out_mask)
    tl.store(indices_ptr + batch_base_n * indices_stride_n,
             best_idx,
             mask=out_mask)

    # Release lock
    tl.atomic_xchg(lock, 0)


def nm_dist(xyz1: torch.Tensor, xyz2: torch.Tensor):
    assert xyz1.shape[-1] == xyz2.shape[-1], "Incompatible dimensions"
    assert xyz1.is_contiguous(), "Matrix xyz1 must be contiguous"
    assert xyz2.is_contiguous(), "Matrix xyz2 must be contiguous"

    N, D = xyz1.shape
    M, D = xyz2.shape

    dists = torch.inf * torch.ones((N, ), device=xyz1.device, dtype=xyz1.dtype)
    indices = torch.zeros((N, ), device=xyz1.device, dtype=torch.int32)
    lock = torch.zeros((N, ), device=xyz1.device, dtype=torch.int32)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']),
                         triton.cdiv(M, META['BLOCK_SIZE_M']))

    nm_dist_kernel[grid](xyz1, xyz2, dists, indices, lock, N, M,
                         xyz1.stride(0), xyz1.stride(1), xyz2.stride(0),
                         xyz2.stride(1), dists.stride(0), indices.stride(0),
                         lock.stride(0))

    return dists, indices


if __name__ == "__main__":

    torch.manual_seed(0)

    xyz1 = torch.randn(17, 3).cuda()
    xyz2 = torch.randn(33, 3).cuda()

    dist1, idx1 = nm_dist(xyz1, xyz2)
    dist2, idx2 = nm_dist(xyz2, xyz1)
    dist1_ref, idx1_ref, dist2_ref, idx2_ref = closest_neighbour_sp(
        xyz1.cpu().numpy(),
        xyz2.cpu().numpy())

    ic(np.isclose(dist1.cpu(), dist1_ref).sum() == len(xyz1))
    ic(np.isclose(idx1.cpu(), idx1_ref).sum() == len(xyz1))
    ic(np.isclose(dist2.cpu(), dist2_ref).sum() == len(xyz2))
    ic(np.isclose(idx2.cpu(), idx2_ref).sum() == len(xyz2))

    exit()

    num_pts = 400000

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        xyz1 = torch.randn(num_pts, 3).cuda()
        xyz1.requires_grad_(True)
        xyz2 = torch.randn(num_pts, 3).cuda()

        with record_function("Triton"):
            triton_out = nm_dist(xyz1, xyz2)
            triton_out2 = nm_dist(xyz2, xyz1)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    exit()

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["N", "M"],
            x_vals=[4 * np.power(10, i) for i in range(1, 6)],
            line_arg="provider",
            line_vals=["Triton", "MashCPP"],
            line_names=["Triton", "MashCPP"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="NMDist Performance",
            args={}))

    def chamfer_triton(xyz1, xyz2):
        d1, i1 = nm_dist(xyz1, xyz2)
        d2, i2 = nm_dist(xyz2, xyz1)
        return d1, i1, d2, i2

    @triton.testing.perf_report(configs)
    def benchmark(M, N, provider):
        xyz1 = torch.randn(M, 3).cuda()
        xyz2 = torch.randn(N, 3).cuda()
        quantiles = [0.5, 0.2, 0.8]
        if provider == "MashCPP":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mash_cpp.toChamferDistance(xyz1, xyz2),
                quantiles=quantiles)
        if provider == "Triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: chamfer_triton(xyz1, xyz2), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)

    exit()

    num_pts = 400000

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        xyz1 = torch.randn(num_pts, 3).cuda()
        xyz2 = torch.randn(num_pts, 3).cuda()

        with record_function("Triton"):
            nm_dist(xyz1, xyz2)[1]

        with record_function("MashCPP"):
            mash_cpp.toChamferDistance(xyz1, xyz2)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    exit()

    torch.manual_seed(0)

    A = torch.randn((64, 64)).cuda()
    B = torch.randn((64, 64)).cuda()

    C_triton = matmul(A, B)

    A = torch.randn((64, 64)).cuda()
    B = torch.randn((64, 64)).cuda()

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("Triton"):
            C_triton = matmul(A, B)
        with record_function("Torch"):
            C_torch = A @ B

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    ic(torch.allclose(C_triton, C_torch))

    # ic(torch.isclose(C_triton, C_torch))
    exit()

    import closest_neighbour
    from torch.profiler import profile, record_function, ProfilerActivity

    np.random.seed(0)
    nr = 40000
    nq = 40000
    a = np.random.rand(nr, 3)
    b = np.random.rand(nq, 3)

    a = torch.from_numpy(a).float().cuda()
    b = torch.from_numpy(b).float().cuda()

    ic(nr, nq)

    # c = extension_cpp.mymul(a, b)

    import mash_cpp

    # ic(extension_cpp.ops.mymuladd)
    idx = 0

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("Chamfer"):
            torch.ops.extension_cpp.closest_neighbour(a, b)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
