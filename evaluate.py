import json
import random
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
import argparse
from dataclasses import dataclass
import numpy as np

from tome_sam.build_tome_sam import tome_sam_model_registry
from tome_sam.utils import misc
from tome_sam.utils.dataloader import ReadDatasetInput, get_im_gt_name_dict, create_dataloaders, Resize
from tome_sam.utils.tome_presets import SAMToMeSetting


def compute_iou_and_boundary_iou(preds, target) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    @param preds: torch.Tensor(Batch_size, masks per image, H, W)
    @param target: torch.Tensor(Batch_size, masks per image, H, W)

    Return:
    (mask_iou, boundary_iou): Tuple[torch.Tensor(float), torch.Tensor(float)]
    """
    assert target.shape[1] == 1, 'only support one mask per image now'
    if preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]:
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    ious = 0
    boundary_ious = 0
    for i in range(0,len(preds)):
        ious = ious + misc.mask_iou(postprocess_preds[i],target[i])
        boundary_ious = boundary_ious + misc.boundary_iou(target[i], postprocess_preds[i])
    return (ious / len(preds)).item(), (boundary_ious / len(preds)).item()


@dataclass
class EvaluateArgs:
    dataset: str
    output: str
    model_type: str
    checkpoint: str
    device: str
    seed: int
    input_size: List[int]
    batch_size: int
    multiple_masks: bool
    num_masks: int = None
    tome_setting: Optional[SAMToMeSetting] = None


def evaluate(args: EvaluateArgs = None):
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    ### Start evaluation ###
    print(f"--- Start evaluation ---")
    test_stats = {}
    metric_logger = misc.MetricLogger(delimiter="  ")
    print(f"valid dataloader length: {len(valid_dataloader)}")


    for data_val in metric_logger.log_every(valid_dataloader, 200):
        imidx, inputs, labels, shapes, labels_ori = data_val["imidx"], data_val["image"], data_val["label"], data_val["shape"], data_val["ori_label"]

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_ori = labels_ori.to(device)

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
            # batched output - list([dict(['masks', 'iou_predictions', 'low_res_logits'])])
            # masks - (image=1, masks per image, H, W)
            batched_output = tome_sam(batched_input, multimask_output=False)

        pred_masks = torch.tensor(np.array([output['masks'][0].cpu() for
                                            output in batched_output])).float().to(device)
        mask_iou, boundary_iou = compute_iou_and_boundary_iou(pred_masks, labels_ori)
        loss_dict = {"mask_iou": mask_iou, "boundary_iou": boundary_iou}
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        metric_logger.update(**loss_dict_reduced)

    print('============================')
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    test_stats.update(resstat)

    return test_stats





def get_args_parser():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of the segmentation result")

    parser.add_argument('--dataset', choices=dataset_name_mapping.keys(), type=str,
                        required=True, help='Specify one of the available datasets: {}'
                        .format(", ".join(dataset_name_mapping.keys())))
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the directory where masks and evaluation results will be stored')
    parser.add_argument('--model_type', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b',
                        help='The type of SAM model to load')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the SAM checkpoint to use for mask generation')
    parser.add_argument('--device', type=str, default="mps",
                        help='The device to run generation on.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', nargs=2, default=[1024, 1024], type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--multiple_masks', action='store_true',
                        help='Enable multiple mask outputs. Require --num_masks to be specified.')
    parser.add_argument('--num_masks', type=int, required=True,
                        help='Specify the number of masks to output (only if --multiple_masks is set).')
    parser.add_argument(
        "--tome_setting",
        type=str,
        default=None,  # Default to None if not provided
        help="JSON string for ToMe settings (e.g., '{\"0\": {\"kv_mode\": {\"r\": 0.6, \"sx\": 2, \"sy\": 2}, \"q_mode\": {\"r\": 0.8, \"sx\": 4, \"sy\": 4}}, ...}')"
    )

    return parser


def parse_and_convert_args() -> EvaluateArgs:
    parser = get_args_parser()
    args = parser.parse_args()

    tome_setting: Optional[SAMToMeSetting] = None

    # Parse the JSON string into a dictionary if provided
    if args.tome_setting is not None:
        try:
            tome_setting = json.loads(args.tome_setting)
            print("Parsed ToMe Settings:", tome_setting)
        except json.JSONDecodeError as e:
            print(f"Error parsing ToMe settings: {e}")
    else:
        print("No ToMe settings provided. Proceeding with default behavior.")

    if args.multiple_masks and args.num_masks is None:
        parser.error("--num_masks must be specified if --multiple_masks is set.")

    evaluate_args = EvaluateArgs(
        dataset=args.dataset,
        output=args.output,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        seed=int(args.seed),
        input_size=args.input_size,
        batch_size=int(args.batch_size),
        multiple_masks=args.multiple_masks,
        num_masks=args.num_masks,
        tome_setting=tome_setting,
    )

    return evaluate_args



# valid set
dataset_coift_val = ReadDatasetInput(
    name="COIFT",
    im_dir="./data/thin_object_detection/COIFT/images",
    gt_dir="./data/thin_object_detection/COIFT/masks",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_hrsod_val = ReadDatasetInput(
    name="HRSOD",
    im_dir="./data/thin_object_detection/HRSOD/images",
    gt_dir="./data/thin_object_detection/HRSOD/masks_max255",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_thin_val = ReadDatasetInput(
    name="ThinObject5k-TE",
    im_dir="./data/thin_object_detection/ThinObject5K/images_test",
    gt_dir="./data/thin_object_detection/ThinObject5K/masks_test",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_dis_val = ReadDatasetInput(
    name="DIS5K-VD",
    im_dir="./data/DIS5K/DIS-VD/im",
    gt_dir="./data/DIS5K/DIS-VD/gt",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_name_mapping = {
    "dis": dataset_dis_val,
    "thin": dataset_thin_val,
    "hrsod": dataset_hrsod_val,
    "coift": dataset_coift_val
}

def parse_and_convert_args() -> EvaluateArgs:
    parser = get_args_parser()
    args = parser.parse_args()

    tome_setting: Optional[SAMToMeSetting] = None

    # Parse the JSON string into a dictionary if provided
    if args.tome_setting is not None:
        try:
            tome_setting = json.loads(args.tome_setting)
            print("Parsed ToMe Settings:", tome_setting)
        except json.JSONDecodeError as e:
            print(f"Error parsing ToMe settings: {e}")
    else:
        print("No ToMe settings provided. Proceeding with default behavior.")

    if args.multiple_masks and args.num_masks is None:
        parser.error("--num_masks must be specified if --multiple_masks is set.")

    evaluate_args = EvaluateArgs(
        dataset=args.dataset,
        output=args.output,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        seed=int(args.seed),
        input_size=args.input_size,
        batch_size=int(args.batch_size),
        multiple_masks=args.multiple_masks,
        num_masks=args.num_masks,
        tome_setting=tome_setting,
    )

    return evaluate_args



# valid set
dataset_coift_val = ReadDatasetInput(
    name="COIFT",
    im_dir="./data/thin_object_detection/COIFT/images",
    gt_dir="./data/thin_object_detection/COIFT/masks",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_hrsod_val = ReadDatasetInput(
    name="HRSOD",
    im_dir="./data/thin_object_detection/HRSOD/images",
    gt_dir="./data/thin_object_detection/HRSOD/masks_max255",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_thin_val = ReadDatasetInput(
    name="ThinObject5k-TE",
    im_dir="./data/thin_object_detection/ThinObject5K/images_test",
    gt_dir="./data/thin_object_detection/ThinObject5K/masks_test",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_dis_val = ReadDatasetInput(
    name="DIS5K-VD",
    im_dir="./data/DIS5K/DIS-VD/im",
    gt_dir="./data/DIS5K/DIS-VD/gt",
    im_ext=".jpg",
    gt_ext=".png"
)

dataset_name_mapping = {
    "dis": dataset_dis_val,
    "thin": dataset_thin_val,
    "hrsod": dataset_hrsod_val,
    "coift": dataset_coift_val
}

if __name__ == "__main__":
    args = parse_and_convert_args()
    evaluate(args)