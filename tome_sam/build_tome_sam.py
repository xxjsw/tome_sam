from dataclasses import dataclass
from functools import partial
from typing import Optional, List


import torch

from segment_anything.modeling import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer
from tome_sam.tome_image_encoder import ToMeImageEncoderViT
from tome_sam.utils.tome_presets import SAMToMeSetting


# Aggregation of SAM image encoder parameters which can be easily accessed by other code snippets
@dataclass
class SAMImageEncoderConfig:
    """
    Configuration class for SAM image encoder
    """
    model_type: str # (vit-b, vit-l, vit-h)
    depth: int
    image_size: int
    vit_patch_size: int
    embed_dim: int
    num_heads: int
    window_size: int
    global_attn_indexes: List[int]

SAM_CONFIGS = {
    'vit-b': SAMImageEncoderConfig(
        model_type='vit-b',
        depth=12,
        image_size=1024,
        vit_patch_size=16,
        embed_dim=768,
        num_heads=12,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
    ),
    'vit-l': SAMImageEncoderConfig(
        model_type='vit-l',
        depth=24,
        image_size=1024,
        vit_patch_size=16,
        embed_dim=1024,
        num_heads=16,
        window_size=14,
        global_attn_indexes=[5, 11, 17, 23],
    ),
    'vit-h': SAMImageEncoderConfig(
        model_type='vit-h',
        depth=32,
        image_size=1024,
        vit_patch_size=16,
        embed_dim=1280,
        num_heads=16,
        window_size=14,
        global_attn_indexes=[7, 15, 23, 31],
    ),
}

def get_sam_config(model_type: str) -> SAMImageEncoderConfig:
    return SAM_CONFIGS[model_type]



def _build_tome_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        tome_setting: Optional[SAMToMeSetting] = None,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ToMeImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            tome_setting=tome_setting,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def build_tome_sam_vit_h(checkpoint=None, tome_setting: Optional[SAMToMeSetting] = None):
    return _build_tome_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        tome_setting=tome_setting,
    )


build_sam = build_tome_sam_vit_h


def build_tome_sam_vit_l(checkpoint=None, tome_setting: Optional[SAMToMeSetting] = None):
    return _build_tome_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        tome_setting=tome_setting,
    )


def build_tome_sam_vit_b(checkpoint=None, tome_setting: Optional[SAMToMeSetting] = None):
    return _build_tome_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        tome_setting=tome_setting
    )


tome_sam_model_registry = {
    "default": build_tome_sam_vit_h,
    "vit_h": build_tome_sam_vit_h,
    "vit_l": build_tome_sam_vit_l,
    "vit_b": build_tome_sam_vit_b,
}