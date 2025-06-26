import numpy as np
from scipy.spatial.distance import cdist

import jax
from jax import numpy as jnp, jit, vmap
from jax.experimental import pallas as pl
from functools import partial
import time

from icecream import ic


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
    batch_n_mask = (batch_base_n < N)[:, None]

    batch_base_m = base_m + jnp.arange(0, BLOCK_SIZE_M)
    batch_m_mask = (batch_base_m < M)[:, None]

    xyz1_x = pl.load(
        xyz1_ref,
        (pl.dslice(None), pl.dslice(0, 1)),
        mask=batch_n_mask,
        other=jnp.inf,
    )

    xyz1_y = pl.load(
        xyz1_ref,
        (pl.dslice(None), pl.dslice(1, 1)),
        mask=batch_n_mask,
        other=jnp.inf,
    )

    xyz1_z = pl.load(
        xyz1_ref,
        (pl.dslice(None), pl.dslice(2, 1)),
        mask=batch_n_mask,
        other=jnp.inf,
    )

    xyz2_x = pl.load(
        xyz2_ref,
        (pl.dslice(None), pl.dslice(0, 1)),
        mask=batch_m_mask,
        other=-jnp.inf,
    )

    xyz2_y = pl.load(
        xyz2_ref,
        (pl.dslice(None), pl.dslice(1, 1)),
        mask=batch_m_mask,
        other=-jnp.inf,
    )

    xyz2_z = pl.load(
        xyz2_ref,
        (pl.dslice(None), pl.dslice(2, 1)),
        mask=batch_m_mask,
        other=-jnp.inf,
    )

    x2 = xyz1_x[:, None, :] - xyz2_x[None, :, :]
    y2 = xyz1_y[:, None, :] - xyz2_y[None, :, :]
    z2 = xyz1_z[:, None, :] - xyz2_z[None, :, :]
    d = x2 * x2 + y2 * y2 + z2 * z2

    # jnp.min does not return indices
    best_d = jnp.min(d, axis=1)
    best_idx = jnp.argmin(d, axis=1) + base_m

    pl.store(
        dist_ref, (pl.dslice(None), pl.dslice(pid_m, 1)), best_d, mask=batch_n_mask
    )
    pl.store(
        idx_ref, (pl.dslice(None), pl.dslice(pid_m, 1)), best_idx, mask=batch_n_mask
    )


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
            pl.BlockSpec((BLOCK_SIZE_M, D), lambda i, j: (0, j)),
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
def _chamfer_distance_pallas(xyz1, xyz2, BLOCK_SIZE_M_1, BLOCK_SIZE_M_2):
    dist1, idx1 = vmap(nmdist_pallas, in_axes=(0, 0, None, None))(
        xyz1, xyz2, 16, BLOCK_SIZE_M_1
    )
    dist2, idx2 = vmap(nmdist_pallas, in_axes=(0, 0, None, None))(
        xyz2, xyz1, 16, BLOCK_SIZE_M_2
    )
    return dist1, idx1, dist2, idx2


def chamfer_distance_pallas(xyz1, xyz2):
    return _chamfer_distance_pallas(
        xyz1, xyz2, get_block_dim(xyz2.shape[1]), get_block_dim(xyz1.shape[1])
    )


if __name__ == "__main__":
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    xyz1 = jax.random.normal(key1, (10000, 1600, 3))
    xyz2 = jax.random.normal(key2, (10000, 1000, 3))

    chamfer_distance_pallas(xyz1, xyz2)

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
