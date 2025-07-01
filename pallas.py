import numpy as np
from scipy.spatial.distance import cdist

import jax
from jax import numpy as jnp, jit, vmap
from jax.experimental import pallas as pl
from functools import partial
import time

from icecream import ic
import torch.utils.dlpack


@jit
def secondary_reduction(dist, idx):
    min_idx = jnp.argmin(dist, axis=-1)
    dist = jnp.take_along_axis(dist, min_idx[:, None], axis=1)
    idx = jnp.take_along_axis(idx, min_idx[:, None], axis=1)
    return dist.reshape(
        -1,
    ), idx.reshape(
        -1,
    )


@partial(jax.jit, static_argnames=["BLOCK_SIZE_N", "BLOCK_SIZE_M"])
def nmdist_kernel_pallas(
    xyz1_ref,
    xyz2_ref,
    dist_ref,
    idx_ref,
    N: int,
    M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_M: int,
):
    pid_n = pl.program_id(axis=0)
    pid_m = pl.program_id(axis=1)

    base_n = pid_n * BLOCK_SIZE_N
    base_m = pid_m * BLOCK_SIZE_M

    batch_base_n = base_n + jnp.arange(0, BLOCK_SIZE_N)
    batch_n_mask = batch_base_n < N

    batch_base_m = base_m + jnp.arange(0, BLOCK_SIZE_M)
    batch_m_mask = batch_base_m < M

    xyz1_x = xyz1_ref[:, 0]
    xyz1_y = xyz1_ref[:, 1]
    xyz1_z = xyz1_ref[:, 2]

    xyz1_x = jnp.where(batch_n_mask, xyz1_x, jnp.inf)
    xyz1_y = jnp.where(batch_n_mask, xyz1_y, jnp.inf)
    xyz1_z = jnp.where(batch_n_mask, xyz1_z, jnp.inf)

    xyz2_x = xyz2_ref[:, 0]
    xyz2_y = xyz2_ref[:, 1]
    xyz2_z = xyz2_ref[:, 2]

    xyz2_x = jnp.where(batch_m_mask, xyz2_x, -jnp.inf)
    xyz2_y = jnp.where(batch_m_mask, xyz2_y, -jnp.inf)
    xyz2_z = jnp.where(batch_m_mask, xyz2_z, -jnp.inf)

    x2 = xyz1_x[:, None] - xyz2_x[None, :]
    y2 = xyz1_y[:, None] - xyz2_y[None, :]
    z2 = xyz1_z[:, None] - xyz2_z[None, :]
    d = x2 * x2 + y2 * y2 + z2 * z2

    # jnp.min does not return indices
    best_d = jnp.min(d, axis=1)
    best_idx = jnp.argmin(d, axis=1) + base_m

    dist_ref[:, pid_m] = best_d
    idx_ref[:, pid_m] = best_idx


@partial(jax.jit, static_argnames=["BLOCK_SIZE_N", "BLOCK_SIZE_M"])
def nmdist_pallas(xyz1, xyz2, BLOCK_SIZE_N, BLOCK_SIZE_M):
    N, D = xyz1.shape
    M, D = xyz2.shape

    kernel = partial(
        nmdist_kernel_pallas,
        N=N,
        M=M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    dist, idx = pl.pallas_call(
        kernel,
        grid=(pl.cdiv(N, BLOCK_SIZE_N), pl.cdiv(M, BLOCK_SIZE_M)),
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE_N, D), lambda i, j: (i, 0)),
            pl.BlockSpec((BLOCK_SIZE_M, D), lambda i, j: (j, 0)),
        ],
        out_specs=[
            pl.BlockSpec((BLOCK_SIZE_N, 1), lambda i, j: (i, 0)),
            pl.BlockSpec((BLOCK_SIZE_N, 1), lambda i, j: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((N, pl.cdiv(M, BLOCK_SIZE_M)), jnp.float32),
            jax.ShapeDtypeStruct((N, pl.cdiv(M, BLOCK_SIZE_M)), jnp.int32),
        ],
    )(xyz1, xyz2)
    return secondary_reduction(dist, idx)


@partial(jax.jit, static_argnames=["BLOCK_SIZE_N", "BLOCK_SIZE_M"])
def batched_nmdist_pallas(xyz1, xyz2, BLOCK_SIZE_N, BLOCK_SIZE_M):
    return vmap(nmdist_pallas, in_axes=(0, 0, None, None))(
        xyz1, xyz2, BLOCK_SIZE_N, BLOCK_SIZE_M
    )


def test_performance(cfg, N, M, n=10):
    data = [
        {
            "xyz1": jax.random.normal(jax.random.PRNGKey(i), (10000, N, 3)),
            "xyz2": jax.random.normal(jax.random.PRNGKey(i + 1), (10000, M, 3)),
        }
        for i in range(n)
    ]

    # warmup
    _ = (batched_nmdist_pallas(**data[0], **cfg)[-1].block_until_ready(),)

    start_time = time.time()

    for i in range(n):
        batched_nmdist_pallas(**data[i], **cfg)[-1].block_until_ready()

    pallas_time = time.time() - start_time
    average_time = pallas_time / n
    print(f"Average time: {average_time:.5f}s", cfg)
    return average_time


def cdiv(x: int, y: int):
    return (x + y - 1) // y


def get_block_dim(M):
    block_dim_m = np.array([256, 512, 1024, 2048])
    tails = cdiv(M, block_dim_m)
    idx = np.argmin(tails)
    return block_dim_m[idx].item()


@partial(jax.jit, static_argnames=["BLOCK_SIZE_M_1", "BLOCK_SIZE_M_2"])
def chamfer_distance_pallas(xyz1, xyz2, BLOCK_SIZE_M_1, BLOCK_SIZE_M_2):
    dist1, idx1 = nmdist_pallas(xyz1, xyz2, 16, BLOCK_SIZE_M_1)
    dist2, idx2 = nmdist_pallas(xyz2, xyz1, 16, BLOCK_SIZE_M_2)
    return dist1, idx1, dist2, idx2


@jax.custom_vjp
def chamfer_distance_jax(xyz1, xyz2):
    dist1, idx1, dist2, idx2 = chamfer_distance_pallas(
        xyz1, xyz2, get_block_dim(xyz1.shape[0]), get_block_dim(xyz2.shape[0])
    )
    return (dist1, idx1, dist2, idx2)


def chamfer_distance_fwd_jax(xyz1: jax.Array, xyz2: jax.Array):
    dist1, idx1, dist2, idx2 = chamfer_distance_jax(xyz1, xyz2)
    return (dist1, idx1, dist2, idx2), (xyz1, idx1, xyz2, idx2)


@jit
def chamfer_distance_bwd_jax(res, dz):
    xyz1, idx1, xyz2, idx2 = res
    dz1, _, dz2, _ = dz

    d_dist1 = dz1[:, None] * 2 * (xyz1 - xyz2[idx1])
    d_dist2 = dz2[:, None] * 2 * (xyz2 - xyz1[idx2])

    dxyz1 = d_dist1.at[idx2, :].add(-d_dist2)
    dxyz2 = d_dist2.at[idx1, :].add(-d_dist1)

    return (dxyz1, dxyz2)


chamfer_distance_jax.defvjp(chamfer_distance_fwd_jax, chamfer_distance_bwd_jax)


if __name__ == "__main__":
    import torch
    from batch_chamfer import nm_dist

    torch.manual_seed(0)

    xyz1 = torch.randn(2, 16, 3).cuda()
    xyz2 = torch.randn(2, 15, 3).cuda()

    import mash_cpp

    _, dist2_mashcpp, _, idx2_mashcpp = mash_cpp.toChamferDistance(xyz1, xyz2)
    dist2_triton, idx2_triton = nm_dist(xyz2, xyz1)

    xyz1 = jax.dlpack.from_dlpack(xyz1)
    xyz2 = jax.dlpack.from_dlpack(xyz2)

    dist2_jax, idx2_jax = vmap(nmdist_pallas, in_axes=(0, 0, None, None))(
        xyz2, xyz1, 16, 16
    )

    ic(idx2_triton[1])
    ic(idx2_mashcpp[1])
    ic(idx2_jax[1])

    dist2_jax, idx2_jax = nmdist_pallas(xyz2[1], xyz1[1], 16, 16)
    ic(idx2_jax)

    exit()

    cfgs = [
        {"BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_M": BLOCK_SIZE_M}
        for BLOCK_SIZE_N in [16, 32, 64, 128]
        for BLOCK_SIZE_M in [256, 512, 1024, 2048]
    ]

    ts = []
    for cfg in cfgs:
        t = test_performance(cfg, 1600, 1000)
        ts.append(t)

    best_idx = np.argmin(jnp.array(ts))
    print("==============")
    print("Best", ts[best_idx], cfgs[best_idx])

    ts = []
    for cfg in cfgs:
        t = test_performance(cfg, 1000, 1600)
        ts.append(t)

    best_idx = np.argmin(jnp.array(ts))
    print("==============")
    print("Best", ts[best_idx], cfgs[best_idx])
