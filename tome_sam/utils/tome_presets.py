from dataclasses import dataclass
from typing import Dict, Union

@dataclass
class ToMe: # settings required to do tome
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
    mode: str # 'tome' or 'pitome' or 'tomesd'
    params: Union[ToMe, PiToMe, ToMeSD]


# key - index of the ViT layer, value - the specific tomesd settings taken place in this block
SAMToMeSetting = Dict[int, ToMeConfig]
