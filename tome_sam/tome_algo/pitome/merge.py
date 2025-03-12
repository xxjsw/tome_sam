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


def generate_src_and_dst_idx(energy_score: torch.Tensor,
                             sx: int=2,
                             sy: int=2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Divide the grid into small cells(sx*sy), and choose the one with the largest energy score as the dst tokens,
    and the left ones as src tokens.

    Args:
        energy_score: records energy_score for each input token (B, H, W)
        sx: stride in the x dimension for choosing dst
        sy: stride in the y dimension for choosing dst

    Returns:
        src_idx, dst_idx: two disjoint sets for src and dst token indices, respectively
    """

    B, H, W = energy_score.shape
    assert H % sy == 0 and W % sx == 0, 'sy must devide H and sx must devide W'

    hsy, hsw = H // sy, W // sx
    num_dst = hsy * hsw

    score_cells = energy_score.view(B, hsy, sy, hsw, sx)
    score_cells = score_cells.permute(0, 1, 3, 2, 4).contiguous().view(B, num_dst, sy*sx)

    _, max_idx = score_cells.max(dim=-1) # min gradient in each cell

    grid_row_indices = torch.arange(hsy, device=energy_score.device).repeat_interleave(torch.tensor(hsw, device=energy_score.device)) # (H*W) flattened row indices for each cell
    grid_col_indices = torch.arange(hsw, device=energy_score.device).repeat(torch.tensor(hsy, device=energy_score.device)) # (H*W) flattened column indices for each cell

    max_local_row_idx = max_idx // sx
    max_local_col_idx = max_idx % sx

    max_global_row_idx = grid_row_indices * sy + max_local_row_idx # (B, num_dst)
    max_global_col_idx = grid_col_indices * sx + max_local_col_idx # (B, num_dst)

    dst_idx = max_global_row_idx * W + max_global_col_idx # (B, num_dst)

    # Mark all positions as 0 first
    idx_buffer = torch.zeros(B, H*W, device=energy_score.device, dtype=torch.int64) # (B, H*W)
    # Mark all dst positions as -1, the left src positions are still 0
    idx_buffer.scatter_(dim=-1, index=dst_idx, src=-torch.ones_like(dst_idx)) # (B, num_dst)
    # sorting idx buffer gives us dst|src and we have src_idx accordingly
    src_idx = idx_buffer.argsort(dim=-1)[:, num_dst:] # (B, H*W - num_dst)

    return src_idx, dst_idx


def pitome(
        metric=None,
        scores: torch.Tensor = None,
        sim: torch.Tensor = None,
        r: int = 0,
) -> Tuple[Callable, Callable]:
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    B, T, _ = metric.shape
    H = W = int(T ** 0.5)
    a_idx, b_idx = generate_src_and_dst_idx(scores.view(B, H, W)) # (B, T-num_dst), (B, num_dst)

    num_dst = b_idx.shape[1]
    # get pairwise similarity between src set(a_idx) and dst set(b_idx)
    sim = gather(sim, dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, num_dst))
    sim = gather(sim, dim=-2, index=a_idx.unsqueeze(-1).expand(B, T - num_dst, num_dst))

    # can't reduce more than the # tokens in src
    r = min(T - num_dst, r)
    node_max, node_idx = sim.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

    unm_idx = edge_idx[..., r:, :]  # lower pairwise similarity - unmerged tokens
    src_idx = edge_idx[..., :r, :]  # higher pairwise similarity - merged tokens
    dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def split(x):
        C = x.shape[-1]
        src = gather(x, dim=1, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[1], C))
        dst = gather(x, dim=1, index=b_idx.unsqueeze(-1).expand(B, b_idx.shape[1], C))

        return src, dst

    def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        merged_tokens = torch.cat([unm, dst], dim=1)

        # To find out indices w.r.t input tensor x, above unm_idx and src_idx are w.r.t src(a_idx).
        # (B, N_unm)
        unm_absolute_indices = gather(a_idx, dim=1, index=unm_idx.squeeze(-1))
        # dst_absolute_indices - (B, N_dst)
        absolute_indices = torch.cat([unm_absolute_indices, b_idx], dim=1)

        return merged_tokens, absolute_indices

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        _, _, c = x.shape
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))
        # Combine back to the original shape
        out = torch.zeros(B, T, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.unsqueeze(-1).expand(B, b_idx.shape[1], c), src=dst)
        out.scatter_(dim=-2,
                     index=gather(a_idx, dim=1, index=unm_idx.squeeze(-1)).unsqueeze(-1).expand(B, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2, index=gather(a_idx, dim=1, index=src_idx.squeeze(-1)).unsqueeze(-1).expand(B, r, c),
                     src=src)

        return out

    return merge, unmerge


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
        merged_tokens = torch.cat([unm, dst], dim=1)

        # To find out indices w.r.t input tensor x, above unm_idx and src_idx are w.r.t src, dst_idx is w.r.t dst
        # (B*num_heads, N_unm)
        unm_absolute_indices = gather(a_idx.unsqueeze(-1), dim=1, index=unm_idx).squeeze(-1)
        absolute_indices = torch.cat([unm_absolute_indices, b_idx], dim=1)

        return merged_tokens, absolute_indices


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        _, _, c = x.shape
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
        margin: torch.Tensor = 0.5, # for thresholding energy score
        alpha=1.0, # for ELU activation
        use_bsm_pitome=False,
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
        energy_score = F.elu((sim - margin), alpha=alpha).mean(dim=-1) # (B, T)
        indices = torch.argsort(energy_score, descending=True) # (B, T)
        # seperate protected token and mergeable tokens
        if use_bsm_pitome:
            return pitome_bsm(metric=metric, indices=indices, scores=sim, r=r)
        else:
            return pitome(metric=metric, scores=energy_score, sim=sim, r=r)



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

