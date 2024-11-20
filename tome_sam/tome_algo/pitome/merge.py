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

def safe_normalize(x: torch.Tensor, dim: int=-1, eps: float=1e-12):
    """
    Safely normalize a tensor by handling zero vectors.
    Args:
        x: Input tensor of shape (B, N, C)
        dim: Dimension along which to normalize
        eps: epsilon to prevent division by zero
    Returns:
        Normalized tensor with same shape as input
    """
    norm = x.norm(dim=dim, keepdim=True)
    zero_mask = (norm < eps).squeeze(dim)
    norm = norm.clamp(min=eps)
    x = x / norm
    x[zero_mask] = 0.0

    return x


def do_nothing(x, mode=None):
    return x


def pitome(
        metric=None,
        indices: torch.Tensor = None,
        scores: torch.Tensor = None,
        r: int = None
) -> Tuple[Callable, Callable]:
    B, T, T = scores.shape
    merge_idx = indices[..., :2 * r]
    protected_idx = indices[..., 2 * r:]
    a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]

    # get similarity scores between mergeable tokens
    scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r))
    scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r))
    _, dst_idx = scores.max(dim=-1)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:

        B, T, C = x.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]

        dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)


def pitome_bsm(
        metric=None,
        indices: torch.Tensor = None,
        scores: torch.Tensor = None,
        r: int = None
) -> Tuple[Callable, Callable]:
    with torch.no_grad():
        B, T, T = scores.shape
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2]
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1]))
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        pass


    return merge, unmerge


def pitome_vision(
        metric: torch.Tensor,
        ratio: float = 1.0,
        margin: torch.Tensor = 0.5,
        alpha=1.0,
        use_bsm_pitome=True
):
    with torch.no_grad():
        B, T, C = metric.shape
        if ratio < 1.0:
            r = math.floor(T - T * ratio)
        else:
            return do_nothing, do_nothing

        # calculate energy score
        metric = F.normalize(metric, p=2, dim=-1)
        sim = metric @ metric.transpose(-1, -2)
        energy_score = F.elu((sim - margin), alpha=alpha).mean(dim=-1)
        indices = torch.argsort(energy_score, descending=True)
        # seperate protected token and mergeable tokens
        if use_bsm_pitome:
            return pitome_bsm(metric=metric, indices=indices, scores=sim, r=r)
        else:
            return pitome(metric=metric, indices=indices, scores=sim, r=r)



def merge_mean(
        merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="mean")
    return x


def prune(
        merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="prune")
    return x


def merge_wavg(
        merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size


def merge_source(
        merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def merge_attention_mask(
        merge, attention_mask: torch.Tensor
):
    attention_mask = merge(attention_mask, mode="amax")
    return attention_mask

