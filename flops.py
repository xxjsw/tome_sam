import random

import numpy as np
import torch
from tqdm import tqdm

from evaluate import parse_and_convert_args, EvaluateArgs, dataset_name_mapping
from tome_sam.build_tome_sam import tome_sam_model_registry
from tome_sam.utils import misc
from tome_sam.utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
from fvcore.nn import FlopCountAnalysis
from functools import partial


def get_flops(args: EvaluateArgs) -> float:
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gflops = []

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    ### Create eval dataloader ###
    print(f"--- Create valid dataloader with dataset {args.dataset} ---")
    dataset_info = dataset_name_mapping[args.dataset]
    valid_im_gt_path = get_im_gt_name_dict(dataset_info, flag='valid')
    valid_dataloader, valid_dataset = create_dataloaders(valid_im_gt_path,
                                                         my_transforms=[
                                                             Resize(args.input_size),
                                                         ],
                                                         batch_size=args.batch_size,
                                                         training=False)
    print(f"--- Valid dataloader with dataset {args.dataset} created ---")

    ### Create model with specified arguments ###
    print(f"--- Create SAM {args.model_type} with token merging in layers {args.tome_setting} ---")

    tome_sam = tome_sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        tome_setting=args.tome_setting,
    )
    tome_sam.to(device)
    tome_sam.eval()

    ### Start Flop count analysis ###
    print(f"--- Start flop count analysis ---")
    for data_val in tqdm(valid_dataloader, position=0, leave=False):
        imidx, inputs, labels, shapes, labels_ori = data_val["imidx"], data_val["image"], data_val["label"], data_val["shape"], data_val["ori_label"]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # (B, C, H, W) -> (B, H, W, C)
        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

        labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
        batched_input = []

        for b_i in range(len(imgs)):
            dict_input = dict()
            input_image = torch.as_tensor(imgs[b_i].astype(np.uint8), device=device).permute(2, 0, 1).contiguous() # (C, H, W)
            dict_input['image'] = input_image
            dict_input['boxes'] = labels_box[b_i: b_i + 1]
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.extend([dict_input])

        with torch.no_grad():
            flops = FlopCountAnalysis(tome_sam, (batched_input, False))
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            gflops.append((flops.total()/1e9)/args.batch_size)

    return np.mean(gflops)


if __name__ == "__main__":
    args = parse_and_convert_args()
    flops_per_image = get_flops(args)
    print(f'Average flops per image: {flops_per_image}')