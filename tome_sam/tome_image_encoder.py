# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from collections.abc import Callable

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type, Dict

from tome_sam.tome_algo.tome.merge import bipartite_soft_matching_random2d

from segment_anything.modeling.image_encoder import Attention, ImageEncoderViT
from .common import LayerNorm2d, MLPBlock
from .tome_algo.pitome.merge import pitome_vision
from .utils.tome_presets import SAMToMeSetting, ViTToMeConfig


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ToMeImageEncoderViT(ImageEncoderViT):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            tome_setting: Optional[SAMToMeSetting] = None,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            tome_setting(Dict[int, ViTToMe]): specify which layers to do token merging and the specific bsm tome parameters
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        if tome_setting is None:
            tome_setting = dict()

        self.blocks = nn.ModuleList()

        for i in range(depth):
            if i in tome_setting:
                vit_tome_param = tome_setting[i]
            else:
                vit_tome_param = None

            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                tome_setting=vit_tome_param
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
            tome_setting: ViTToMeConfig = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
            tome_setting: if it is not None, allow token merging(use EfficientAttention rather than normal Attention)
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        if tome_setting:
            self.attn = EfficientAttention(
                tome_setting=tome_setting,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size if window_size == 0 else (window_size, window_size),
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size if window_size == 0 else (window_size, window_size),
            )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x



class EfficientAttention(Attention):
    def __init__(
            self,
            tome_setting: ViTToMeConfig,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, use_rel_pos, rel_pos_zero_init, input_size)
        self.tome_setting = tome_setting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X - (B, H, W, C * nHeads)
        B, H, W, _ = x.shape
        C = _ // self.num_heads

        # X - (B * nHeads, N, C)
        x = x.reshape(B, H, W, self.num_heads, C).permute(0, 3, 1, 2, 4).reshape(B * self.num_heads, H * W, C)

        x_q = x
        x_kv = x

        # token merging on q
        x_merge, x_unmerge = Callable, Callable
        if self.tome_setting.q.mode == 'bsm':
            x_merge, x_unmerge = bipartite_soft_matching_random2d(
                metric=x_q, w=W, h=H,
                r=int(H * W * self.tome_setting.q.params.r),
                sx=self.tome_setting.q.params.sx, sy=self.tome_setting.q.params.sy,
                no_rand=True
            )

        if self.tome_setting.q.mode == 'pitome':
            x_merge, x_unmerge = pitome_vision(
                metric=x_q, ratio=self.tome_setting.q.params.r,
                margin=torch.tensor(self.tome_setting.q.params.margin),
                alpha=self.tome_setting.q.params.alpha,
            )

        # x_q_idx - (B*num_heads, Nq_reduced), which records the order of tokens w.r.t original input tensor
        x_q, x_q_idx = x_merge(x_q)

        _, Nq_reduced, _ = x_q.shape
        # print("Nq_reduced", Nq_reduced)
        # reshape x_q from (B*nHeads, Nq_reduced, C) to (B, Nq_reduced, C*nHeads)
        x_q = x_q.reshape(B, self.num_heads, Nq_reduced, C).permute(0, 2, 1, 3).reshape(B, Nq_reduced, C*self.num_heads)
        # qkv in shape of (B, Nq_reduced, C*nHeads*3)
        qkv = self.qkv(x_q)
        # qkv in shape of (3, B, nHeads, Nq_reduced, C)
        qkv = qkv.reshape(B, Nq_reduced, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        # q in shape of (B*nHeads, Nq_reduced, C)
        q, _, _ = qkv.reshape(3, B*self.num_heads, Nq_reduced, C).unbind(0)

        # token merging on kv
        kv_merge = Callable
        if self.tome_setting.kv.mode == 'bsm':
            kv_merge, _, = bipartite_soft_matching_random2d(
                metric=x_kv, w=W, h=H,
                r=int(H * W * self.tome_setting.kv.params.r),
                sx=self.tome_setting.kv.params.sx, sy=self.tome_setting.kv.params.sy,
                no_rand=True
            )

        if self.tome_setting.kv.mode == 'pitome':
            kv_merge, _ = pitome_vision(
                metric=x_kv, ratio=self.tome_setting.kv.params.r,
                margin=torch.tensor(self.tome_setting.kv.params.margin),
                alpha=self.tome_setting.kv.params.alpha,
            )

        x_kv, x_kv_idx = kv_merge(x_kv)

        _, Nkv_reduced, _ = x_kv.shape
        # reshape x_kv from (B*nHeads, Nkv_reduced, C) to (B, Nkv_reduced, C*nHeads)
        x_kv = x_kv.reshape(B, self.num_heads, Nkv_reduced, C).permute(0, 2, 1, 3).reshape(B, Nkv_reduced, C*self.num_heads)
        # qkv in shape of (B, Nkv_reduced, C*nHeads*3)
        qkv = self.qkv(x_kv)
        # qkv in shape of (3, B, nHeads, Nkv_reduced, C)
        qkv = qkv.reshape(B, Nkv_reduced, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        # q in shape of (B*nHeads, Nkv_reduced, C)
        _, k, v = qkv.reshape(3, B*self.num_heads, Nkv_reduced, C).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)


        if self.use_rel_pos:
           attn = add_decomposed_rel_pos(attn, self.rel_pos_h, self.rel_pos_w,
                                         (H, W), (H, W), x_q_idx, x_kv_idx)

        attn = attn.softmax(dim=-1)
        x = attn @ v
        # token unmerge
        x = x_unmerge(x)
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def get_2d_indices(idx: torch.Tensor,width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 1D indices to 2D (row, col) indices for a grid of given width.
    """
    row_idx = idx // width
    col_idx = idx % width
    return row_idx, col_idx


def batched_index_and_sum(Rh, q_row_idx, k_row_idx):
    """
    Index `Rh` based on `q_row_idx` and `k_row_idx` in a memory-efficient way, then sum over the last dimension.

    Args:
        Rh (torch.Tensor): Tensor of shape (B, q_h, k_h, c).
        q_row_idx (torch.Tensor): Tensor of shape (B, n1).
        k_row_idx (torch.Tensor): Tensor of shape (B, n2).

    Returns:
        torch.Tensor: Tensor of shape (B, n1, n2), summing over the last dimension of `Rh`.
    """
    B, q_h, k_h, c = Rh.shape
    _, n1 = q_row_idx.shape
    _, n2 = k_row_idx.shape

    # Prepare output tensor
    output = torch.zeros(B, n1, n2, device=Rh.device, dtype=Rh.dtype)

    # Batch-wise computation
    for b in range(B):
        # Get batch-specific tensors
        q_idx = q_row_idx[b]  # Shape (n1,)
        k_idx = k_row_idx[b]  # Shape (n2,)

        # Index into Rh for this batch
        gathered = Rh[b, q_idx][:, k_idx]  # Shape (n1, n2, c)

        # Sum over the last dimension (channels)
        output[b] = gathered.sum(dim=-1)  # Shape (n1, n2)

    return output


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Because we do token merging, this method needs an update. As after merging the most similar tokens, the remaining
    token sequence does not have a regular grid size. We have the indices of merged token sequence w.r.t input tensor x,
    so we extract the remaining relative position embeddings from the original ones based on these indices

    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
        q_idx (Tensor): index of the merged query tokens w.r.t original query tokens.
        k_idx (Tensor): index of the merged key tokens w.r.t original key tokens.

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """

    B, N_q, N_k = attn.shape
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    # Broadcast Rh and Rw to have batch dimension
    Rh = Rh.unsqueeze(0).expand(B, -1, -1, -1) # (B, q_h, k_h, c)
    Rw = Rw.unsqueeze(0).expand(B, -1, -1, -1) # (B, q_w, k_w, c)

    # print("Rh", Rh.shape, Rh)
    # print("Rw", Rw.shape, Rw)
    # print("attn", attn.shape)

    # Convert 1D indices to 2D(h-w) indices for queries and keys
    q_row_idx, q_col_idx = get_2d_indices(q_idx, q_w)
    k_row_idx, k_col_idx = get_2d_indices(k_idx, k_w)
    # print("q_row_idx", q_row_idx.shape)
    # print("k_row_idx", k_row_idx.shape)
    print("1")
    rel_pos_embedding_h = batched_index_and_sum(Rh, q_row_idx, k_row_idx)
    print("2")
    rel_pos_embedding_w = batched_index_and_sum(Rw, q_col_idx, k_col_idx)
    print("3")
    # print("rel_pos_embedding_h", rel_pos_embedding_h.shape)
    # print("rel_pos_embedding_w", rel_pos_embedding_w.shape)

    # Add relative position embeddings to attention map
    attn = attn + rel_pos_embedding_h + rel_pos_embedding_w

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x