# Temporary file with conversion utils. Will be removed in the final version of the PR

import re
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn as tnn


###############################################################################
#                              dtype conversions                              #
###############################################################################

def pt2jax_dtype(pt_dtype: torch.dtype):
    if not isinstance(pt_dtype, torch.dtype):
        raise ValueError(f"The argument to to_jax_dtype() must be an instance of " +
                         f"torch.dtype, but instead {type(pt_dtype)} was received")
    # not using dicts because dtypes don't have stable hash
    if pt_dtype == torch.float32:
        return jnp.float32
    if pt_dtype == torch.float16:
        return jnp.float16
    if pt_dtype == torch.bfloat16:
        return jnp.bfloat16
    else:
        raise ValueError(f"Converting {pt_dtype} to a JAX type is not implemented")


def jax2pt_dtype(dtype: jnp.dtype):
    if not isinstance(dtype, jnp.dtype):
        raise ValueError(f"The argument to to_pytorch_dtype() must be an instance of " +
                         f"jnp.dtype, but instead {type(dtype)} was received")
    # not using dicts because dtypes don't have stable hash
    if dtype == jnp.float32:
        return torch.float32
    if dtype == jnp.float16:
        return torch.float16
    if dtype == jnp.bfloat16:
        return torch.bfloat16
    else:
        raise ValueError(f"Converting {dtype} to a PyTorch type is not implemented")


###############################################################################
#                              array conversions                              #
###############################################################################

def jax2pt(x: jax.Array):
    if x.dtype == jnp.bfloat16:
        # convert via fp32 because numpy doesn't support bf16
        x32 = x.astype(jnp.float32)
        return torch.from_numpy(np.asarray(x32).copy()).bfloat16()
    else:
        return torch.from_numpy(np.asarray(x).copy())


def pt2jax(pt_x: torch.Tensor):
    if pt_x.dtype == torch.bfloat16:
        # convert via fp32 because numpy doesn't support bf16
        pt_x32 = pt_x.to(torch.float32)
        return jnp.array(pt_x32.detach().numpy(), dtype=jnp.bfloat16)
    else:
        return jnp.array(pt_x.detach().numpy())
