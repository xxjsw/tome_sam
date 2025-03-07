# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

# Adapt from https://github.com/hchautran/PiToMe/blob/main/algo/pitome/merge.py, remove class token, add unmerge part because
# our task is segmentation.

import math
from typing import Callable, Tuple
import torch
import torch.nn.functional as F

def get_central_difference_gradient(x: torch.Tensor) -> torch.Tensor:
    '''
    Compute the gradient magnitude for token embeddings using central difference approximation.

    Args:
        x: token embeddings of shape (B, H, W, C)

    Returns:
        grad_mag(torch.Tensor): Gradient magnitude of embeddings (B, H, W)
    '''
    padded = F.pad(x, (0, 0, 1, 1, 1, 1), mode='replicate') # (B, H+2, W+2)
    diff_x = (padded[:, 1:-1, 2:, :] - padded[:, 1:-1, :-2, :])/2.0 # Difference along width # (B, H, W, C)
    diff_y = (padded[:, 2:, 1:-1, :] - padded[:, :-2, 1:-1, :])/2.0 # Difference along height # (B, H, W, C)
    grad_mag = torch.sqrt((diff_x ** 2).sum(dim=-1) + (diff_y ** 2).sum(dim=-1)) # (B, H, W)

    return grad_mag

def get_sobel_gradient(x: torch.Tensor) -> torch.Tensor:
    '''
    Compute the gradient magnitude for token embeddings using a Sobel operator.

    Args:
        x: token embeddings of shape (B, H, W, C)

    Returns:
        grad_mag(torch.Tensor): Gradient magnitude of embeddings (B, H, W)
    '''
    x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
    B, C, H, W = x.shape

    sobel_x = torch.tensor([[-1., 0, 1],
                            [-2., 0, 2],
                            [-1., 0, 1]], device=x.device, dtype=x.dtype)

    sobel_y = torch.tensor([[-1., -2, 1],
                            [0, 0, 0],
                            [1., 2, 1]], device=x.device, dtype=x.dtype)

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C, 1, 3, 3)

    grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)  # (B, C, H, W)
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)  # (B, C, H, W)

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=1)  # (B, H, W)

    return grad_mag


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def do_nothing(x, mode=None):
    return x


def grad_bipartite_soft_matching(metric: torch.Tensor,
                                 r: int = 0) -> Tuple[Callable, Callable]: # r - number of tokens to remove
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        B, T, C = metric.shape
        if r <= 0:
            return do_nothing, do_nothing

        H = W = int(T**0.5)
        grad = get_sobel_gradient(metric.view(B, H, W, C)) # (B, H, W)
        # grad = get_central_difference_gradient(metric.view(B, H, W, C))  # (B, H, W)
        grad = grad.view(B, H*W) # (B, N)
        indices = torch.argsort(grad, descending=False) # (small gradient - less informative)

        # seperate protected token and mergeable tokens
        merge_idx = indices[..., :2 * r]
        protected_idx = indices[..., 2 * r:]
        a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]  # src and dst

        def split(x):
            _, _, C = x.shape
            src = gather(x, dim=1, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[1], C))
            dst = gather(x, dim=1, index=b_idx.unsqueeze(-1).expand(B, b_idx.shape[1], C))
            return src, dst

        metric = F.normalize(metric, p=2, dim=-1)
        a, b = split(metric)
        sim = a @ b.transpose(-1, -2)

        _, dst_idx = sim.max(dim=-1) # dst_idx (B, r), for each src token records the index of the dst token, w.r.t b_idx


    def merge(x: torch.Tensor, mode='mean') -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        protected = x[batch_idx, protected_idx, :]
        src, dst = split(x)
        dst = dst.scatter_reduce(dim=-2, index=dst_idx.unsqueeze(-1).expand(B, r, C), src=src, reduce=mode)

        merged_tokens = torch.cat([protected, dst], dim=1)
        absolute_indices = torch.cat([protected_idx, b_idx], dim=1)

        return merged_tokens, absolute_indices


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        _, _, c = x.shape
        protected, dst = x[:, :-r, :], x[:, -r:, :]
        src = gather(dst, dim=-2, index=dst_idx.unsqueeze(-1).expand(B, r, c))

        out = torch.zeros(B, T, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=protected_idx.unsqueeze(-1).expand(B, protected_idx.shape[1], c), src=protected)
        out.scatter_(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, c), src=src)
        out.scatter_(dim=-2, index=b_idx.unsqueeze(-1).expand(B, r, c), src=dst)

        return out

    return merge, unmerge





