# Copyright 2020 Google LLC
# Copyright 2025 Anonymous authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(__file__)

# ---- CPU extension (required) ----
isotonic_cpu_ext = load(
    name="isotonic_cpu_ext",
    sources=[os.path.join(_THIS_DIR, "isotonic_cpu.cpp")],
    verbose=False,
)

# Expose CPU functions from the extension
isotonic_kl_cpu = isotonic_cpu_ext.isotonic_kl
isotonic_kl_backward_cpu = isotonic_cpu_ext.isotonic_kl_backward
isotonic_l2_cpu = isotonic_cpu_ext.isotonic_l2
isotonic_l2_backward_cpu = isotonic_cpu_ext.isotonic_l2_backward

# ---- CUDA extension (optional) ----
def _cuda_unavailable(*args, **kwargs):
    raise RuntimeError(
        "CUDA isotonic ops requested but CUDA extension is not available "
        "(either torch.cuda.is_available() is False or the build failed)."
    )

isotonic_kl_cuda = _cuda_unavailable
isotonic_kl_backward_cuda = _cuda_unavailable
isotonic_l2_cuda = _cuda_unavailable
isotonic_l2_backward_cuda = _cuda_unavailable

if torch.cuda.is_available():
    try:
        isotonic_cuda_ext = load(
            name="isotonic_cuda_ext",
            sources=[os.path.join(_THIS_DIR, "isotonic_cuda.cu")],
            verbose=False,
        )
        isotonic_kl_cuda = isotonic_cuda_ext.isotonic_kl
        isotonic_kl_backward_cuda = isotonic_cuda_ext.isotonic_kl_backward
        isotonic_l2_cuda = isotonic_cuda_ext.isotonic_l2
        isotonic_l2_backward_cuda = isotonic_cuda_ext.isotonic_l2_backward
    except Exception as e:
        print("[soft_binary_arg_max_ops] WARNING: CUDA extension failed to build:", e)
        print("[soft_binary_arg_max_ops] Falling back to CPU-only isotonic ops.")

def get_hypersimplex_basis(n, k, *, device=None, dtype=None):
    assert 0 <= k <= n
    x = torch.zeros(n, device=device, dtype=dtype)
    x[:k] = 1
    return x

def soft_binary_argmax_k(values, k, regularization_strength=1.0):
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    
    n = values.shape[1]
    hypersimplex_basis = get_hypersimplex_basis(n, k, device=values.device)

    return soft_binary_argmax.apply(values,hypersimplex_basis, "l2", regularization_strength)


isotonic_l2 = {"cpu": isotonic_l2_cpu, "cuda": isotonic_l2_cuda}
isotonic_kl = {"cpu": isotonic_kl_cpu, "cuda": isotonic_kl_cuda}
isotonic_l2_backward = {
    "cpu": isotonic_l2_backward_cpu,
    "cuda": isotonic_l2_backward_cuda,
}
isotonic_kl_backward = {
    "cpu": isotonic_kl_backward_cpu,
    "cuda": isotonic_kl_backward_cuda,
}


def _arange_like(x, reverse=False):
    # returns arange with len of x of the same dtype and device (assumes 2d, first dim batch)
    if reverse:
        ar = torch.arange(x.shape[1] - 1, -1, -1, dtype=x.dtype, device=x.device)
    else:
        ar = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
    return ar.expand(x.shape[0], -1)

def _inv_permutation(permutation):
    # returns inverse permutation of 'permutation'. (assumes 2d, first dim batch)
    inv_permutation = torch.zeros_like(permutation)
    inv_permutation.scatter_(1, permutation, _arange_like(permutation))
    return inv_permutation


# The following is from google-research/fast-soft-sort with the following modifications:
# - replace numpy functions with torch equivalent
# - remove uncessary operations
# - reimplement backward pass in C++

class soft_binary_argmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, polytope_basis, regularization="l2", regularization_strength=1.0,):
        ctx.scale = 1.0 / regularization_strength
        ctx.regularization = regularization
        w = polytope_basis
        theta = tensor * ctx.scale
        s, permutation = torch.sort(theta, descending=True)
        inv_permutation = _inv_permutation(permutation)
        if ctx.regularization == "l2":
            dual_sol = isotonic_l2[s.device.type](s - w)
            ret = (s - dual_sol).gather(1, inv_permutation)
            factor = torch.tensor(1.0, device=s.device)
        else:
            dual_sol = isotonic_kl[s.device.type](s, torch.log(w))
            ret = torch.exp((s - dual_sol).gather(1, inv_permutation))
            factor = ret

        ctx.save_for_backward(factor, s, dual_sol, permutation, inv_permutation)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        factor, s, dual_sol, permutation, inv_permutation = ctx.saved_tensors
        grad = (grad_output * factor).clone()
        if ctx.regularization == "l2":
            grad -= isotonic_l2_backward[s.device.type](
                s, dual_sol, grad.gather(1, permutation)
            ).gather(1, inv_permutation)
        else:
            grad -= isotonic_kl_backward[s.device.type](
                s, dual_sol, grad.gather(1, permutation)
            ).gather(1, inv_permutation)
        return grad * ctx.scale, None, None, None