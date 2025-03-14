import torch
import jax
from jax import numpy as jnp, vmap, jit
import mash_cpp
import jax_triton as jt

from chamfer_loss import nm_dist_kernel

import polyscope as ps
from icecream import ic



if __name__ == "__main__":
    torch.manual_seed(0)

    ic(jt)

    anchor_num = 800
    mask_boundary_sample_num = 90

    mask_boundary_phi_idxs = torch.repeat_interleave(torch.arange(anchor_num), mask_boundary_sample_num).cuda()
    boundary_pts = torch.randn((anchor_num * mask_boundary_sample_num, 3))



