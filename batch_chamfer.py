import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.distance import cdist
from typing import List

import triton
import triton.language as tl

from icecream import ic


@triton.jit
def nm_dist_kernel2(
    xyz1_ptr,
    xyz2_ptr,
    dists_ptr,
    indices_ptr,
    B,
    N,
    M,
    xyz1_stride_d,
    xyz1_stride_b,
    xyz1_stride_n,
    xyz2_stride_d,
    xyz2_stride_b,
    xyz2_stride_m,
    dist_stride_b,
    dist_stride_n,
    indices_stride_b,
    indices_stride_n,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    base_b = pid_b * BLOCK_SIZE_B
    base_n = pid_n * BLOCK_SIZE_N

    batch_base_b = base_b + tl.arange(0, BLOCK_SIZE_B)
    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)

    batch_n_mask = (batch_base_b[:, None] < B) & (batch_base_n[None, :] < N)

    xyz1_x = tl.load(
        xyz1_ptr
        + batch_base_b[:, None] * xyz1_stride_b
        + batch_base_n[None, :] * xyz1_stride_n,
        mask=batch_n_mask,
        other=100,
    )
    xyz1_y = tl.load(
        xyz1_ptr + N * B
        + batch_base_b[:, None] * xyz1_stride_b
        + batch_base_n[None, :] * xyz1_stride_n,
        mask=batch_n_mask,
        other=100,
    )
    xyz1_z = tl.load(
        xyz1_ptr + 2 * (N * B)
        + batch_base_b[:, None] * xyz1_stride_b
        + batch_base_n[None, :] * xyz1_stride_n,
        mask=batch_n_mask,
        other=100,
    )

    cur_best_d = tl.full((BLOCK_SIZE_B, BLOCK_SIZE_N), float("inf"), tl.float32)
    cur_best_idx = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), tl.int32)

    for base_m in tl.range(0, M, BLOCK_SIZE_M):
        batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
        batch_m_mask = (batch_base_b[:, None] < B) & (batch_base_m[None, :] < M)

        xyz2_x = tl.load(
            xyz2_ptr
            + batch_base_b[:, None] * xyz2_stride_b
            + batch_base_m[None, :] * xyz2_stride_m,
            mask=batch_m_mask,
            other=-100,
        )
        xyz2_y = tl.load(
            xyz2_ptr + (M * B)
            + batch_base_b[:, None] * xyz2_stride_b
            + batch_base_m[None, :] * xyz2_stride_m,
            mask=batch_m_mask,
            other=-100,
        )
        xyz2_z = tl.load(
            xyz2_ptr + 2 * (M * B)
            + batch_base_b[:, None] * xyz2_stride_b
            + batch_base_m[None, :] * xyz2_stride_m,
            mask=batch_m_mask,
            other=-100,
        )

        x2 = xyz1_x[:, :, None] - xyz2_x[:, None, :]
        y2 = xyz1_y[:, :, None] - xyz2_y[:, None, :]
        z2 = xyz1_z[:, :, None] - xyz2_z[:, None, :]
        d = x2 * x2 + y2 * y2 + z2 * z2

        best_d = tl.min(d, axis=2)

        # ​TODO: sqrt depends on SLU. Let pytorch handle it for now
        # best_d = tl.sqrt(best_d)
        best_idx = tl.argmin(d, axis=2) + base_m

        mask = best_d < cur_best_d
        cur_best_d = tl.where(mask, best_d, cur_best_d)
        cur_best_idx = tl.where(mask, best_idx, cur_best_idx)

    tl.store(
        dists_ptr
        + batch_base_b[:, None] * dist_stride_b
        + batch_base_n[None, :] * dist_stride_n,
        cur_best_d,
        mask=batch_n_mask,
    )
    tl.store(
        indices_ptr
        + batch_base_b[:, None] * indices_stride_b
        + batch_base_n[None, :] * indices_stride_n,
        cur_best_idx,
        mask=batch_n_mask,
    )


def nm_dist2(xyz1: torch.Tensor, xyz2: torch.Tensor):
    xyz1 = xyz1.permute(2, 0, 1).contiguous()
    xyz2 = xyz2.permute(2, 0, 1).contiguous()

    D, B, N = xyz1.shape
    D, B, M = xyz2.shape

    dists = torch.zeros((B, N), device=xyz1.device, dtype=xyz1.dtype)
    indices = torch.zeros((B, N), device=xyz1.device, dtype=torch.int32)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_SIZE_B"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    configs = {
        "BLOCK_SIZE_B": 1,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_M": 512,
        "num_warps": 2,
        "num_stages": 3,
    }

    nm_dist_kernel2[grid](
        xyz1,
        xyz2,
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
        **configs,
    )
    return dists, indices


@torch.library.custom_op("mash::chamfer_distance", mutates_args=())
def chamfer_distance(xyz1: torch.Tensor, xyz2: torch.Tensor) -> List[torch.Tensor]:
    dist1, idx1 = nm_dist2(xyz1, xyz2)
    dist2, idx2 = nm_dist2(xyz2, xyz1)
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

    xyz1 = torch.randn(10000, 1536, 3).cuda()
    xyz1.requires_grad_(True)
    xyz2 = torch.randn(10000, 1024, 3).cuda()
    xyz2.requires_grad_(True)

    dist1, idx1, dist2, idx2 = chamfer_distance(xyz1, xyz2)
    exit()

    dist1_mashcpp, dist2_mashcpp, idx1_mashcpp, idx2_mashcpp = (
        mash_cpp.toChamferDistance(xyz1, xyz2)
    )

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
