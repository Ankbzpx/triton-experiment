import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.distance import cdist
from typing import List
import jax
import torch.utils
import torch.utils.dlpack
from pallas import chamfer_distance_jax

import triton
import triton.language as tl

from icecream import ic


def get_cuda_autotune_config():
    return (
        [
            triton.Config(
                {
                    "BLOCK_SIZE_B": 1,
                    "BLOCK_SIZE_N": BLOCK_SIZE_N,
                    "BLOCK_SIZE_M": int(16384 / BLOCK_SIZE_N),
                },
                num_warps=num_warps,
                num_stages=3,
            )
            for BLOCK_SIZE_N in [16, 32, 64]
            for num_warps in [2, 4, 8]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_B": 1,
                    "BLOCK_SIZE_N": BLOCK_SIZE_N,
                    "BLOCK_SIZE_M": int(32768 / BLOCK_SIZE_N),
                },
                num_warps=num_warps,
                num_stages=3,
            )
            for BLOCK_SIZE_N in [16, 32, 64]
            for num_warps in [2, 4, 8]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_B": 1,
                    "BLOCK_SIZE_N": BLOCK_SIZE_N,
                    "BLOCK_SIZE_M": int(65536 / BLOCK_SIZE_N),
                },
                num_warps=num_warps,
                num_stages=3,
            )
            for BLOCK_SIZE_N in [16, 32, 64]
            for num_warps in [2, 4, 8]
        ]
    )


# @triton.autotune(configs=get_cuda_autotune_config(), key=["B", "N", "M"])
@triton.jit
def nm_dist_kernel(
    xyz1_ptr,
    xyz2_ptr,
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
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    base_b = pid_b * BLOCK_SIZE_B
    base_n = pid_n * BLOCK_SIZE_N

    batch_base_b = base_b + tl.arange(0, BLOCK_SIZE_B)
    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)

    batch_n_mask = (batch_base_b[:, None] < B) & (batch_base_n[None, :] < N)

    xyz1_x_ptr = tl.make_block_ptr(
        base=xyz1_ptr,
        shape=(B, N),
        strides=(xyz1_stride_b, xyz1_stride_n),
        offsets=(base_b, base_n),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_N),
        order=(1, 0),
    )
    xyz1_y_ptr = tl.make_block_ptr(
        base=xyz1_ptr + 1,
        shape=(B, N),
        strides=(xyz1_stride_b, xyz1_stride_n),
        offsets=(base_b, base_n),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_N),
        order=(1, 0),
    )
    xyz1_z_ptr = tl.make_block_ptr(
        base=xyz1_ptr + 2,
        shape=(B, N),
        strides=(xyz1_stride_b, xyz1_stride_n),
        offsets=(base_b, base_n),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_N),
        order=(1, 0),
    )

    xyz1_x = tl.load(xyz1_x_ptr)
    xyz1_y = tl.load(xyz1_y_ptr)
    xyz1_z = tl.load(xyz1_z_ptr)

    xyz1_x = tl.where(batch_n_mask, xyz1_x, float("inf"))
    xyz1_y = tl.where(batch_n_mask, xyz1_y, float("inf"))
    xyz1_z = tl.where(batch_n_mask, xyz1_z, float("inf"))

    cur_best_d = tl.full((BLOCK_SIZE_B, BLOCK_SIZE_N), float("inf"), tl.float32)
    cur_best_idx = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), tl.int32)

    xyz2_x_ptr = tl.make_block_ptr(
        base=xyz2_ptr,
        shape=(B, M),
        strides=(xyz2_stride_b, xyz2_stride_m),
        offsets=(base_b, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_M),
        order=(1, 0),
    )
    xyz2_y_ptr = tl.make_block_ptr(
        base=xyz2_ptr + 1,
        shape=(B, M),
        strides=(xyz2_stride_b, xyz2_stride_m),
        offsets=(base_b, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_M),
        order=(1, 0),
    )
    xyz2_z_ptr = tl.make_block_ptr(
        base=xyz2_ptr + 2,
        shape=(B, M),
        strides=(xyz2_stride_b, xyz2_stride_m),
        offsets=(base_b, 0),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_M),
        order=(1, 0),
    )

    for base_m in tl.range(0, M, BLOCK_SIZE_M):
        batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
        batch_m_mask = (batch_base_b[:, None] < B) & (batch_base_m[None, :] < M)

        xyz2_x = tl.load(xyz2_x_ptr)
        xyz2_y = tl.load(xyz2_y_ptr)
        xyz2_z = tl.load(xyz2_z_ptr)

        xyz2_x = tl.where(batch_m_mask, xyz2_x, -float("inf"))
        xyz2_y = tl.where(batch_m_mask, xyz2_y, -float("inf"))
        xyz2_z = tl.where(batch_m_mask, xyz2_z, -float("inf"))

        x2 = xyz1_x[:, :, None] - xyz2_x[:, None, :]
        y2 = xyz1_y[:, :, None] - xyz2_y[:, None, :]
        z2 = xyz1_z[:, :, None] - xyz2_z[:, None, :]
        d = x2 * x2 + y2 * y2 + z2 * z2

        best_d, best_idx = tl.min(d, axis=2, return_indices=True)
        best_idx += base_m

        mask = best_d < cur_best_d
        cur_best_d = tl.where(mask, best_d, cur_best_d)
        cur_best_idx = tl.where(mask, best_idx, cur_best_idx)

        # Increament ptr
        xyz2_x_ptr = tl.advance(xyz2_x_ptr, (0, BLOCK_SIZE_M))
        xyz2_y_ptr = tl.advance(xyz2_y_ptr, (0, BLOCK_SIZE_M))
        xyz2_z_ptr = tl.advance(xyz2_z_ptr, (0, BLOCK_SIZE_M))

    cur_best_d_ptr = tl.make_block_ptr(
        base=dists_ptr,
        shape=(B, N),
        strides=(dist_stride_b, dist_stride_n),
        offsets=(base_b, base_n),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(cur_best_d_ptr, cur_best_d, boundary_check=(0, 1))

    cur_best_idx_ptr = tl.make_block_ptr(
        base=indices_ptr,
        shape=(B, N),
        strides=(indices_stride_b, indices_stride_n),
        offsets=(base_b, base_n),
        block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(cur_best_idx_ptr, cur_best_idx, boundary_check=(0, 1))


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
        triton.cdiv(B, META["BLOCK_SIZE_B"]),
    )

    configs = {
        "BLOCK_SIZE_B": 1,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_M": 1024,
        "num_warps": 2,
        "num_stages": 3,
    }

    compiled_kernel = nm_dist_kernel[grid](
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


def batched_chamfer_distance_jax(xyz1, xyz2):
    dist1_jax, idx1_jax, dist2_jax, idx2_jax = jax.vmap(chamfer_distance_jax)(
        jax.dlpack.from_dlpack(xyz1.detach()), jax.dlpack.from_dlpack(xyz2.detach())
    )
    return (
        torch.utils.dlpack.from_dlpack(dist1_jax),
        torch.utils.dlpack.from_dlpack(idx1_jax),
        torch.utils.dlpack.from_dlpack(dist2_jax),
        torch.utils.dlpack.from_dlpack(idx2_jax),
    )


def eval_chamfer_distance_jax(xyz1, xyz2, loss_func):
    def eval_func(xyz1, xyz2):
        dist1_jax, _, dist2_jax, _ = jax.vmap(chamfer_distance_jax)(xyz1, xyz2)
        return loss_func(dist1_jax, dist2_jax)

    d_xyz1_jax, d_xyz2_jax = jax.grad(eval_func, argnums=(0, 1))(
        jax.dlpack.from_dlpack(xyz1.detach()), jax.dlpack.from_dlpack(xyz2.detach())
    )

    return (
        torch.utils.dlpack.from_dlpack(d_xyz1_jax),
        torch.utils.dlpack.from_dlpack(d_xyz2_jax),
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    xyz1 = torch.randn(10000, 1600, 3).cuda()
    xyz2 = torch.randn(10000, 1000, 3).cuda()

    xyz1.requires_grad_(True)
    xyz2.requires_grad_(True)

    dist1, idx1, dist2, idx2 = chamfer_distance(xyz1, xyz2)
    dist1_jax, idx1_jax, dist2_jax, idx2_jax = batched_chamfer_distance_jax(xyz1, xyz2)

    ic(torch.allclose(dist1, dist1_jax))
    ic(torch.allclose(idx1, idx1_jax))
    ic(torch.allclose(dist2, dist2_jax))
    ic(torch.allclose(idx2, idx2_jax))

    def test_loss(d1, d2):
        return d1.mean() - d2.sum()

    loss = test_loss(dist1, dist2)
    d_xyz1 = gradient(loss, xyz1)
    d_xyz2 = gradient(loss, xyz2)

    d_xyz1_jax, d_xyz2_jax = eval_chamfer_distance_jax(xyz1, xyz2, test_loss)

    ic(torch.allclose(d_xyz1, d_xyz1_jax, atol=1e-6))
    ic(torch.allclose(d_xyz2, d_xyz2_jax, atol=1e-6))
    # exit()

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["B", "N", "M"],
            x_vals=np.arange(0, 6) * 2000,
            line_arg="provider",
            line_vals=["Triton", "Pallas"],
            line_names=["Triton", "Pallas"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="NMDist Performance",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(B, M, N, provider):
        ic(B)
        xyz1 = torch.randn(B, 1600, 3).cuda()
        xyz2 = torch.randn(B, 1000, 3).cuda()
        quantiles = [0.5, 0.2, 0.8]
        if provider == "Pallas":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: batched_chamfer_distance_jax(xyz1, xyz2), quantiles=quantiles
            )
        if provider == "Triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: chamfer_distance(xyz1, xyz2), quantiles=quantiles
            )
        perf = lambda ms: 2 * M * N * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)
