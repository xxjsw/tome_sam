import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from evaluate import parse_and_convert_args, EvaluateArgs, dataset_name_mapping
from tome_sam.build_tome_sam import tome_sam_model_registry
from tome_sam.utils import misc
from tome_sam.utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
from fvcore.nn import FlopCountAnalysis

from tome_sam.utils.json_serialization import convert_to_serializable_dict


def get_flops(args: EvaluateArgs) -> dict:
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # gflops_sam = []
    gflops_image_encoder = []

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

        """
        # flops evaluation on whole sam
        with torch.no_grad():
            # because batched_input is a list of dictionary, not a normally expected tensor input, which is required for flops count
            flops = FlopCountAnalysis(tome_sam, (batched_input, args.multiple_masks))
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            gflops_sam.append((flops.total()/1e9)/args.batch_size)
        """

        tome_sam.to(device)
        image_encoder = tome_sam.image_encoder
        input_images = torch.stack([tome_sam.preprocess(x['image']) for x in batched_input], dim=0).to(device)
        # flops evaluation only on image encoder
        with torch.no_grad():
            flops = FlopCountAnalysis(image_encoder, input_images)
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            gflops_image_encoder.append((flops.total()/1e9)/args.batch_size)

    # sam_flops_per_image = np.mean(gflops_sam)
    image_encoder_flops_per_image = np.mean(gflops_image_encoder)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        filename = os.path.join(args.output, 'flops.json')
        with open(filename, 'w') as f:
            json.dump({
                # 'flops/img(sam)': str(sam_flops_per_image),
                'flops/img(image_encoder)': str(image_encoder_flops_per_image),
                'evaluate_args': convert_to_serializable_dict(args),
            }, f, indent=4, default=str)

    return {
            # 'flops/img(sam)': sam_flops_per_image,
            'flops/img(image_encoder)': image_encoder_flops_per_image
            }


if __name__ == "__main__":
    args = parse_and_convert_args()
    avg_flops = get_flops(args)
    print(avg_flops)