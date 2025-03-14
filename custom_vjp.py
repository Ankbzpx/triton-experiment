import jax
from jax import numpy as jnp, vmap, jit, custom_jvp, grad, jvp
import jax_triton as jt
from chamfer_loss import nm_dist_kernel, chamfer_distance
from jax import custom_vjp
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from icecream import ic


# @jit
def nm_dist_triton(xyz1: jax.Array, xyz2: jax.Array):
    N, D = xyz1.shape
    M, D = xyz2.shape

    dists = jnp.inf * jnp.ones((N), dtype=xyz1.dtype)
    indices = jnp.zeros((N, ), dtype=jnp.int32)
    lock = jnp.zeros((N, ), dtype=jnp.int32)

    dist_shape = jax.ShapeDtypeStruct(shape=(xyz1.shape[0], ),
                                      dtype=xyz1.dtype)
    idx_shape = jax.ShapeDtypeStruct(shape=(xyz1.shape[0], ), dtype=jnp.int32)

    metaparams = dict(BLOCK_SIZE_N=16,
                      BLOCK_SIZE_M=512,
                      GROUP_SIZE=16,
                      num_warps=2,
                      num_stages=3)

    grid = (jt.cdiv(N, metaparams['BLOCK_SIZE_N']),
            jt.cdiv(M, metaparams['BLOCK_SIZE_M']))

    jt.triton_call(xyz1,
                   xyz2,
                   dists,
                   indices,
                   lock,
                   N,
                   M,
                   *jt.strides_from_shape(xyz1.shape),
                   *jt.strides_from_shape(xyz2.shape),
                   *jt.strides_from_shape(dists.shape),
                   *jt.strides_from_shape(indices.shape),
                   *jt.strides_from_shape(lock.shape),
                   kernel=nm_dist_kernel,
                   out_shape=[dist_shape, idx_shape],
                   grid=grid,
                   **metaparams)

    return dists, indices


# @jit
def chamfer_distance_jax(xyz1: jax.Array, xyz2: jax.Array):
    dist1, idx1 = nm_dist_triton(xyz1, xyz2)
    dist2, idx2 = nm_dist_triton(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


@custom_vjp
def f(x, y):
    return jnp.sin(x) * y


def f_fwd(x, y):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return f(x, y), (jnp.cos(x), jnp.sin(x), y)


def f_bwd(res, g):
    cos_x, sin_x, y = res  # Gets residuals computed in f_fwd
    return (cos_x * g * y, sin_x * g)


f.defvjp(f_fwd, f_bwd)

if __name__ == "__main__":

    key_1, key_2 = jax.random.split(jax.random.PRNGKey(0), 2)
    xyz1 = jax.random.normal(key_1, (8196, 3), dtype=jnp.float32)
    xyz2 = jax.random.normal(key_2, (8196, 3), dtype=jnp.float32)

    # ic(chamfer_distance_jax(xyz1, xyz2))

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        
        # with record_function("Jax-Triton"):
        #     chamfer_distance_jax(xyz1, xyz2)
        
        xyz1 = torch.randn(8196, 3).cuda()
        xyz2 = torch.randn(8196, 3).cuda()

        with record_function("Pytorch-Triton"):
            chamfer_distance(xyz1, xyz2)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

