## TOME SAM: Applying BSM token merging strategy onto Segment Anything Model(SAM)

### Installation
Refer to installation steps of the [Segment Anything repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#installation).

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions here to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install `tome_sam` repository and install with:
```
git clone https://github.com/xxjsw/tome_sam.git
cd tome_sam
pip install -e.
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. jupyter is also required to run the example notebooks.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### Download Checkpoints
Create a directory `checkpoints` directly inside the root directory and download the pre-trained SAM checkpoints into it:
```
# ViT-B 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-H 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Prepare Dataset
Refer to dataset preparation from [Segment Anything in high Quality](https://github.com/SysCV/sam-hq/blob/main/train/README.md#1-data-preparation).
Download available datasets from [Hugging Face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data), and organize the
dataset with the following structure:
```
tome_sam
|____data
   |____DIS5K
   |____thin_object_detection
   | |____COIFT
   | |____HRSOD
   | |____ThinObject5K
```
### Inference
`example.py` provides an example of how to run inference with specified token merging setting for each ViT block.
0. Choose the merging methods, including Tome, Tome for diffusion (ToMeSD), and PiToMe at lines ...
1. Define the required token merging parameters in a `dict`, where the key represents the exact layer index and its 
value determines the bsm setting taken place in this ViT block.
```
test_tome_setting: SAMToMeSetting = {
    2: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
    5: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
    8: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
    11: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
}
```

2. Call iou or flops evaluation with all required arguments.
```
from evaluate import EvaluateArgs, evaluate
from flops import get_flops

evaluate_args = EvaluateArgs(
    dataset="dis",
    output="",
    model_type="vit_b",
    checkpoint="checkpoints/sam_vit_b_01ec64.pth",
    device="mps",
    seed=0,
    input_size=[1024, 1024],
    batch_size=1,
    multiple_masks=False,
    tome_setting = test_tome_setting,
)

iou_results = evaluate(evaluate_args) # mask_iou and boundary_iou
flops_per_image = get_flops(evaluate_args)
```

It is also possible to run inference from command line by setting certain flags, please refer to the parser arguments defined in
`evaluate.py`

### Questions

1. I removed the class_token in Pitome. Does it impact to model performance?
2. There is something with zero padding, Do we need to modify normalize function in PiToMe?
   For (2), I will do some modify ..
   
