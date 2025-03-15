import numpy as np
import jax
from jax import numpy as jnp, vmap, jit, custom_jvp, grad, jvp
import jax_triton as jt
from chamfer_loss import nm_dist_kernel, nm_dist, chamfer_distance, closest_neighbour_sp
from jax import custom_vjp
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from icecream import ic


def nm_dist_triton(xyz1: jnp.ndarray, xyz2: jnp.ndarray):
    N, D = xyz1.shape
    M, D = xyz2.shape

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

    xyz1_stride = jt.strides_from_shape(xyz1.shape)
    xyz2_stride = jt.strides_from_shape(xyz2.shape)
    lock_stride = jt.strides_from_shape(lock.shape)

    return jt.triton_call(xyz1,
                          xyz2,
                          lock,
                          N=N,
                          M=M,
                          xyz1_stride_n=xyz1_stride[0],
                          xyz1_stride_d=xyz1_stride[1],
                          xyz2_stride_m=xyz2_stride[0],
                          xyz2_stride_d=xyz2_stride[1],
                          dist_stride_n=lock_stride[0],
                          indices_stride_n=lock_stride[0],
                          lock_stride_n=lock_stride[0],
                          kernel=nm_dist_kernel,
                          out_shape=[dist_shape, idx_shape],
                          grid=grid,
                          **metaparams)


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

    np.random.seed(0)

    xyz1 = np.random.randn(8196, 3).astype(np.float32)
    xyz2 = np.random.randn(8196, 3).astype(np.float32)

    ic(closest_neighbour_sp(xyz1, xyz2))
    ic(chamfer_distance_jax(xyz1, xyz2))
    ic(
        chamfer_distance(
            torch.from_numpy(xyz1).cuda(),
            torch.from_numpy(xyz2).cuda()))
    exit()

    xyz1_torch = torch.randn(8196, 3).cuda()
    xyz1_torch = torch.randn(8196, 3).cuda()

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True) as prof:

        with record_function("Jax-Triton"):
            chamfer_distance_jax(xyz1, xyz2)

        with record_function("Pytorch-Triton"):
            chamfer_distance(xyz1_torch, xyz1_torch)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
