import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.distance import cdist
from typing import List

import triton
import triton.language as tl

from icecream import ic


@triton.jit
def nm_dist_kernel(
    xyz1_ptr,
    xyz2_ptr,
    lock_ptr,
    dists_ptr,
    indices_ptr,
    B,
    N,
    M,
    xyz1_stride_b,
    xyz1_stride_n,
    xyz1_stride_d,
    xyz2_stride_b,
    xyz2_stride_m,
    xyz2_stride_d,
    dist_stride_b,
    dist_stride_n,
    indices_stride_b,
    indices_stride_n,
    lock_stride_b,
    lock_stride_n,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    base_b = pid_b * BLOCK_SIZE_B
    base_n = pid_n * BLOCK_SIZE_N
    base_m = pid_m * BLOCK_SIZE_M

    batch_base_b = base_b + tl.arange(0, BLOCK_SIZE_B)
    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
    batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)

    batch_n_mask = (batch_base_b[:, None] < B) & (batch_base_n[None, :] < N)
    batch_m_mask = (batch_base_b[:, None] < B) & (batch_base_m[None, :] < M)

    d = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N, BLOCK_SIZE_M), tl.float32)
    for i in tl.static_range(3):
        xyz1 = tl.load(
            xyz1_ptr
            + batch_base_b[:, None] * xyz1_stride_b
            + batch_base_n[None, :] * xyz1_stride_n
            + i * xyz1_stride_d,
            mask=batch_n_mask,
            other=100,
        )
        xyz2 = tl.load(
            xyz2_ptr
            + batch_base_b[:, None] * xyz2_stride_b
            + batch_base_m[None, :] * xyz2_stride_m
            + i * xyz2_stride_d,
            mask=batch_m_mask,
            other=-100,
        )
        diff = xyz1[:, :, None] - xyz2[:, None, :]
        d += diff * diff

    best_d = tl.min(d, axis=2)

    # ​TODO: sqrt depends on SLU. Let pytorch handle it for now
    # best_d = tl.sqrt(best_d)
    best_idx = tl.argmin(d, axis=2) + base_m

    lock = lock_ptr + pid_b * lock_stride_b + pid_n * lock_stride_n
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass

    cur_best_d = tl.load(
        dists_ptr
        + batch_base_b[:, None] * dist_stride_b
        + batch_base_n[None, :] * dist_stride_n,
        mask=batch_n_mask,
    )

    # Handle zero initialization in JAX
    # FIXME: The safer option is to use another lock for first occuring pid_n initialization
    out_mask = ((best_d < cur_best_d) | (cur_best_d == 0)) & batch_n_mask
    tl.store(
        dists_ptr
        + batch_base_b[:, None] * dist_stride_b
        + batch_base_n[None, :] * dist_stride_n,
        best_d,
        mask=out_mask,
    )
    tl.store(
        indices_ptr
        + batch_base_b[:, None] * indices_stride_b
        + batch_base_n[None, :] * indices_stride_n,
        best_idx,
        mask=out_mask,
    )

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

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(B, META["BLOCK_SIZE_B"]),
    )

    configs = {
        "BLOCK_SIZE_B": 1,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_M": 512,
        "num_warps": 2,
        "num_stages": 3,
    }

    lock = torch.zeros(
        (
            triton.cdiv(B, configs["BLOCK_SIZE_B"]),
            triton.cdiv(N, configs["BLOCK_SIZE_N"]),
        ),
        device=xyz1.device,
        dtype=torch.int32,
    )

    nm_dist_kernel[grid](
        xyz1,
        xyz2,
        lock,
        dists,
        indices,
        B,
        N,
        M,
        xyz1.stride(0),
        xyz1.stride(1),
        xyz1.stride(2),
        xyz2.stride(0),
        xyz2.stride(1),
        xyz2.stride(2),
        dists.stride(0),
        dists.stride(1),
        indices.stride(0),
        indices.stride(1),
        lock.stride(0),
        lock.stride(1),
        **configs,
    )
    return dists, indices


@torch.library.custom_op("mash::chamfer_distance", mutates_args=())
def chamfer_distance(xyz1: torch.Tensor, xyz2: torch.Tensor) -> List[torch.Tensor]:
    dist1, idx1 = nm_dist(xyz1, xyz2)
    dist2, idx2 = nm_dist(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


def chamfer_distance_setup_context(ctx, inputs, output):
    xyz1, xyz2 = inputs
    _, idx1, _, idx2 = output
    ctx.save_for_backward(xyz1, idx1, xyz2, idx2)


def chamfer_distance_backward(ctx, grad_out):
    xyz1, idx1, xyz2, idx2 = ctx.saved_tensors
    grad_dist1, _, grad_dist2, _ = grad_out

    d1 = xyz1 - torch.gather(xyz2, 1, idx1[..., None].expand(-1, -1, 3).long())
    d2 = xyz2 - torch.gather(xyz1, 1, idx2[..., None].expand(-1, -1, 3).long())

    d_dist1 = grad_dist1[..., None] * 2 * d1
    d_dist2 = grad_dist2[..., None] * 2 * d2

    grad_xyz1 = torch.scatter_add(
        d_dist1, 1, idx2[..., None].expand(-1, -1, 3).long(), -d_dist2
    )
    grad_xyz2 = torch.scatter_add(
        d_dist2, 1, idx1[..., None].expand(-1, -1, 3).long(), -d_dist1
    )
    return grad_xyz1, grad_xyz2


chamfer_distance.register_autograd(
    chamfer_distance_backward, setup_context=chamfer_distance_setup_context
)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


if __name__ == "__main__":
    import mash_cpp

    torch.manual_seed(0)

    # xyz1 = torch.randn(2, 17, 3).cuda()
    # xyz2 = torch.randn(2, 33, 3).cuda()
    # dist1, idx1 = nm_dist(xyz1, xyz2)
    # exit()

    xyz1 = torch.randn(10000, 1600, 3).cuda()
    xyz2 = torch.randn(10000, 1000, 3).cuda()

    xyz1.requires_grad_(True)
    xyz2.requires_grad_(True)

    dist1, idx1, dist2, idx2 = chamfer_distance(xyz1, xyz2)
    exit()

    dist1_mashcpp, dist2_mashcpp, idx1_mashcpp, idx2_mashcpp = (
        mash_cpp.toChamferDistance(xyz1, xyz2)
    )
    # exit()

    def test_loss(d1: torch.Tensor, d2: torch.Tensor):
        return d1.mean() - d2.sum()

    loss = test_loss(dist1, dist2)
    loss_mashcpp = test_loss(dist1_mashcpp, dist2_mashcpp)

    d_xyz1 = gradient(loss, xyz1)
    d_xyz2 = gradient(loss, xyz2)

    d_xyz1_mashcpp = gradient(loss_mashcpp, xyz1)
    d_xyz2_mashcpp = gradient(loss_mashcpp, xyz2)

    ic(torch.allclose(d_xyz1_mashcpp, d_xyz1, atol=1e-6))
    ic(torch.allclose(d_xyz2_mashcpp, d_xyz2, atol=1e-6))

    ic(torch.allclose(dist1, dist1_mashcpp))
    ic(torch.allclose(idx1, idx1_mashcpp))
    ic(torch.allclose(dist2, dist2_mashcpp))
    ic(torch.allclose(idx2, idx2_mashcpp))
    exit()

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["B", "N", "M"],
            x_vals=np.arange(0, 6) * 2000,
            line_arg="provider",
            line_vals=["Triton", "CUDA"],
            line_names=["Triton", "CUDA"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="NMDist Performance",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(B, M, N, provider):
        ic(B)
        xyz1 = torch.randn(B, 1536, 3).cuda()
        xyz2 = torch.randn(B, 1024, 3).cuda()
        quantiles = [0.5, 0.2, 0.8]
        if provider == "CUDA":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: mash_cpp.toChamferDistance(xyz1, xyz2), quantiles=quantiles
            )
        if provider == "Triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: chamfer_distance(xyz1, xyz2), quantiles=quantiles
            )
        perf = lambda ms: 2 * M * N * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)
