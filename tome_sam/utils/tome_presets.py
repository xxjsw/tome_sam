from dataclasses import dataclass
from typing import Dict

@dataclass
class BSMToMe: # settings required to do BSM
    r: float  # Ratio of tokens to be merged
    sx: int   # Stride in the x dimension
    sy: int   # Stride in the y dimension

@dataclass
class ViTToMe: # settings required for one ViT block
    kv_mode: BSMToMe
    q_mode: BSMToMe

# key - index of the ViT layer, value - the specific tome settings taken place in this block
SAMToMeSettings = Dict[int, ViTToMe]

# bsm_hq
tome_cfg = {
    "kv_mode": {
        "kv_r": 0.6,
        "kv_sx": 2,
        "kv_sy": 2,
    },
    "q_mode": {
        "q_r": 0.8,
        "q_sx": 4,
        "q_sy": 4,
    }
}

tome_presets = {
    'bsm_hq': [
        dict(q_mode=None, kv_mode='bsm', kv_r=0.6, kv_sx=2, kv_sy=2),
        dict(q_mode=None, kv_mode='bsm', kv_r=0.6, kv_sx=2, kv_sy=2),
        dict(q_mode='bsm', kv_mode=None, q_r=0.8, q_sx=4, q_sy=4),
        dict(q_mode='bsm', kv_mode=None, q_r=0.8, q_sx=4, q_sy=4)
    ],
    'bsm_fast': [
        dict(q_mode=None, kv_mode='bsm_r2D', kv_r=0.9, kv_sx=4, kv_sy=4),
        dict(q_mode=None, kv_mode='bsm_r2D', kv_r=0.9, kv_sx=4, kv_sy=4),
        dict(q_mode='bsm_r2D', kv_mode=None, q_r=0.9, q_sx=4, q_sy=4),
        dict(q_mode='bsm_r2D', kv_mode=None, q_r=0.9, q_sx=4, q_sy=4)
    ],
    'n2d_2x2': [
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2)),
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2)),
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2)),
        dict(q_mode='neighbor_2D', kv_mode=None, q_s=(2, 2))
    ]
}
