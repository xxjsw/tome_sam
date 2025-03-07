from dataclasses import dataclass
from typing import Dict, Union, Literal, List


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


@dataclass
class ToMe: # settings required to do tome, tome25 or grad_tome
    r: float  # Ratio of tokens to be merged

@dataclass
class ToMeSD: # settings required to do tomesd
    r: float  # Ratio of tokens to be merged
    sx: int   # Stride in the x dimension
    sy: int   # Stride in the y dimension
    no_rand: bool  # if true, disable randomness (use top left corner only)

@dataclass
class PiToMe: # settings required to do pitome
    r: float # Ratio of tokens to be merged
    margin: float # Threshold for energy score
    alpha: float # for ELU activation

@dataclass
class ToMeConfig:
    mode: Literal['tome', 'pitome', 'tomesd', 'tome25', 'grad_tome']
    params: Union[ToMe, PiToMe, ToMeSD]


# key - index of the ViT layer, value - the specific tomesd settings taken place in this block
SAMToMeSetting = Dict[int, ToMeConfig]

def generate_tome_sam_settings(model_type: str, start_idx: int, end_idx: int):
    num_layers = get_sam_config(model_type).depth
    assert start_idx <= end_idx
    assert start_idx < num_layers
    assert end_idx < num_layers

    return {
        'tome': {
            i: ToMeConfig(
                mode='tome',
                params=ToMe(r=0.5)
            ) for i in range(start_idx, end_idx + 1)
        },
        'tome25': {
            i: ToMeConfig(
                mode='tome25',
                params=ToMe(r=0.5)
            ) for i in range(start_idx, end_idx + 1)
        },
        'tomesd(no_random)': {
            i: ToMeConfig(
                mode='tomesd',
                params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
            ) for i in range(start_idx, end_idx + 1)
        },
        'tomesd(random)': {
            i: ToMeConfig(
                mode='tomesd',
                params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=False)
            ) for i in range(start_idx, end_idx + 1)
        },
        'pitome':{
            i: ToMeConfig(
            mode='pitome',
            params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
            ) for i in range(start_idx, end_idx + 1)
        }
    }


# Last 5 layers
LAST_5_LAYERS_TOME = generate_tome_sam_settings('vit-b', 7, 11)

# All Layers
ALL_LAYERS_TOME = generate_tome_sam_settings('vit-b', 0, 11)

