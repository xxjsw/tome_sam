# Adapt from tomesd.merge (https://github.com/dbolya/tomesd/blob/main/tomesd/merge.py)

import torch
from typing import Tuple, Callable

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

def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(
                metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = safe_normalize(metric, dim=-1)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)


    def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        merged_tokens = torch.cat([unm, dst], dim=1)

        # To find out indices w.r.t input tensor x, above unm_idx and src_idx are w.r.t src(a_idx), dst_idx is w.r.t dst(b_idx)
        # (B*num_heads, N_unm)
        unm_absolute_indices = gather(a_idx.expand(n, a_idx.shape[1], 1), dim=1, index=unm_idx).squeeze(-1)
        # (B*num_heads, N_dst)
        dst_absolute_indices = b_idx.squeeze(-1).expand(n, -1)
        absolute_indices = torch.cat([unm_absolute_indices, dst_absolute_indices], dim=1)

        return merged_tokens, absolute_indices


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        _, _, c = x.shape
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))
        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2,
                     index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c),
                     src=src)

        return out

    return merge, unmerge


def random_25_bipartite_soft_matching(metric: torch.Tensor,
                                     r: int,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - r: number of tokens to remove (by merging)
     - generator: random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        rand_idx = torch.rand(B, N, 1, device=metric.device, generator=generator).argsort(dim=1)
        num_dst = int(N*0.25)
        a_idx, b_idx = rand_idx[:, num_dst:, :], rand_idx[:, :num_dst, :]   # src, dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = safe_normalize(metric, dim=-1)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)


    def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        merged_tokens = torch.cat([unm, dst], dim=1)

        # To find out indices w.r.t input tensor x, above unm_idx and src_idx are w.r.t src(a_idx), dst_idx is w.r.t dst(b_idx)
        # (B, N_unm)
        unm_absolute_indices = gather(a_idx.expand(n, a_idx.shape[1], 1), dim=1, index=unm_idx).squeeze(-1)
        # (B, N_dst)
        dst_absolute_indices = b_idx.squeeze(-1).expand(n, -1)
        absolute_indices = torch.cat([unm_absolute_indices, dst_absolute_indices], dim=1)

        return merged_tokens, absolute_indices


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        _, _, c = x.shape
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))
        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2,
                     index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c),
                     src=src)

        return out

    return merge, unmerge