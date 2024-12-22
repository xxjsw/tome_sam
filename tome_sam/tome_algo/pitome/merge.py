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


def pitome(
        metric=None,
        indices: torch.Tensor = None,
        scores: torch.Tensor = None,
        r: int = None
) -> Tuple[Callable, Callable]:

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    B, T, T = scores.shape
    merge_idx = indices[..., :2 * r]
    protected_idx = indices[..., 2 * r:]
    a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]

    # get similarity scores between mergeable tokens
    scores = gather(scores, dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r))
    scores = gather(scores, dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r))
    _, dst_idx = scores.max(dim=-1)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:

        B, T, C = x.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]

        dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)


def pitome_bsm(
        metric=None,
        indices: torch.Tensor = None, # descendingly sorted indices matrix according to the energy score
        scores: torch.Tensor = None,
        r: int = None # number of tokens t be merged
) -> Tuple[Callable, Callable]:

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        B, T, T = scores.shape
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2] # src, dst
        num_dst = b_idx.shape[-1]
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)

        scores = gather(scores, dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1]))

        scores = gather(scores, dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))

        node_max, node_idx = scores.max(dim=-1)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)


    def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        n, t1, c = src.shape
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        merged_tensor = torch.cat([unm, dst], dim=1)

        # To find out indices w.r.t input tensor x, above unm_idx and src_idx are w.r.t src, dst_idx is w.r.t dst
        # (B*num_heads, N_unm)
        unm_absolute_indices = gather(a_idx.unsqueeze(-1), dim=1, index=unm_idx).squeeze(-1)
        absolute_indices = torch.cat([unm_absolute_indices, b_idx], dim=1)
        sorted_indices = absolute_indices.argsort(dim=1)
        merged_tensor = gather(merged_tensor, dim=1, index=sorted_indices.unsqueeze(-1).expand(n, merged_tensor.shape[1], c))

        return merged_tensor, sorted_indices


    def unmerge(x: torch.Tensor, sorted_indices: torch.Tensor) -> torch.Tensor:
        _, _, c = x.shape
        # Compute unsorted_indices from sorted_indices
        unsorted_indices = sorted_indices.argsort(dim=1)  # Indices to reorder sorted_merged_tensor back to "unm + dst"
        # Reorder x back into "unm + dst" structure
        x = gather(x, dim=1, index=unsorted_indices.unsqueeze(-1).expand(x.shape[0], x.shape[1], c))

        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))
        # Combine back to the original shape
        out = torch.zeros(B, T, c, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=b_idx.unsqueeze(-1).expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.unsqueeze(-1).expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.unsqueeze(-1).expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge


def pitome_vision(
        metric: torch.Tensor,
        ratio: float = 0, # ratio of tokens to be merged
        margin: torch.Tensor = 0.5, # for thresholding energy score #TODO: different margins among [0, 1]
        alpha=1.0, # for ELU activation
        use_bsm_pitome=True
):
    with torch.no_grad():
        B, T, C = metric.shape
        if ratio <= 0:
            return do_nothing, do_nothing
        else:
            r = math.floor(T * ratio) # so r means #tokens to be merged

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

