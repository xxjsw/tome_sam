from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class BSMToMe: # settings required to do BSM tome
    r: float  # Ratio of tokens to be merged
    sx: int   # Stride in the x dimension
    sy: int   # Stride in the y dimension

@dataclass
class PiToMe: # settings required to do pitome
    r: float # Ratio of tokens to be merged
    margin: float # Threshold for energy score
    alpha: float # for ELU activation

@dataclass
class ToMeConfig:
    mode: str # 'bsm' or 'pitome'
    params: Union[BSMToMe, PiToMe]


# key - index of the ViT layer, value - the specific tome settings taken place in this block
SAMToMeSetting = Dict[int, ToMeConfig]
