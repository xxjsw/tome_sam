import os
from functools import partial
from typing import Tuple

import cv2
import torch

import argparse
from dataclasses import dataclass
import numpy as np

import matplotlib.pyplot as plt
from segment_anything.utils.transforms import ResizeLongestSide
from tome_sam.build_tome_sam import tome_sam_model_registry
from tome_sam.utils.dataloader import ReadDatasetInput


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()


# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def mask_iou(pred_label: np.ndarray, gt_label: np.ndarray) -> float:
    """
    Calculate mask IoU for binary predicted and ground truth labels.
    """
    intersection = np.logical_and(pred_label, gt_label).sum()
    union = np.logical_or(pred_label, gt_label).sum()

    return intersection / union


def avg_mask_iou(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    assert len(pred_labels) == len(gt_labels)
    ious = []
    for (pred_label, gt_label) in zip(pred_labels, gt_labels):
        ious.append(mask_iou(pred_label, gt_label))

    return sum(ious) / len(ious)


def boundary_iou(pred_label: np.ndarray, gt_label: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """
    Compute boundary IoU between two binary masks by focusing on their edges.
    """
    gt_boundary = mask_to_boundary(gt_label, dilation_ratio)
    dt_boundary = mask_to_boundary(pred_label, dilation_ratio)

    intersection = np.logical_and(gt_boundary, dt_boundary).sum()
    union = np.logical_or(gt_boundary, dt_boundary).sum()

    return intersection / union


def avg_boundary_iou(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    assert len(pred_labels) == len(gt_labels)

    boundary_ious = []
    for (pred_label, gt_label) in zip(pred_labels, gt_labels):
        boundary_ious.append(boundary_iou(pred_label, gt_label))

    return sum(boundary_ious) / len(boundary_ious)


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    """Compute the bounding box around the provided mask.
    The mask should be in format [H, W] where (H, W) are the spatial dimensions.
    Returns a numpy array with the box in xyxy format: [x_min, y_min, x_max, y_max].
    """
    # get bounding box from mask
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    """
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    """
    return np.array([x_min, y_min, x_max, y_max])


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def read_mask(file_path: str) -> np.ndarray:
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask > 128).astype(np.uint8)
    return binary_mask


def select_files_from_dataset(dataset: str, num=20) -> Tuple:
    gt_folder = os.path.join(dataset, 'gt')
    # Check if the folders exist and get the file paths
    files = []
    if os.path.exists(gt_folder):
        files = [os.path.splitext(f)[0] for f in os.listdir(gt_folder)]
        files = files[1: num+1]

    return files


@dataclass
class EvaluateArgs:
    dataset: str
    """
    dataset
    |---gt
    |---im
    """
    output: str
    model_type: str
    checkpoint: str
    device: str
    multiple_masks: bool
    num_masks: int = None
    tome_layers: Tuple[int, ] = ()


def evaluate(args: EvaluateArgs = None):
    """
    if args.multiple_masks and args.num_masks is None:
        logging.info('Error: You must specify --num-masks if --multiple-masks is set.')
        sys.exit(1)

    if args.multiple_masks:
        pass
    else:
        pass
    """

    sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "mps"

    dataset = 'data/DIS5K/DIS-VD'
    filenames = select_files_from_dataset(dataset)

    encoder_depth = 12

    mask_ious_diff_layers = []
    boundary_ious_diff_layers = []

    for layer_i in range(0, encoder_depth+1):
        sam = tome_sam_model_registry[model_type](checkpoint=sam_checkpoint, tome_layers=(layer_i,))
        sam.to(device=device, dtype=torch.float32)
        sam.eval()

        pred_masks = []
        gt_masks = []
        for filename in filenames:
            image = cv2.imread(f'{dataset}/im/{filename}.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bounding_box = mask_to_box(read_mask(f'{dataset}/gt/{filename}.png'))
            boxes = torch.tensor(bounding_box, device=sam.device)

            resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

            batched_input = [
                {
                    'image': prepare_image(image, resize_transform, sam),
                    'boxes': resize_transform.apply_boxes_torch(boxes, image.shape[:2]),
                    'original_size': image.shape[:2]
                },
            ]

            # dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
            batched_output = sam(batched_input, multimask_output=False)
            pred_mask = torch.squeeze(batched_output[0]['masks'][0]).cpu().numpy().astype('uint8')
            pred_masks.append(pred_mask)
            gt_mask = read_mask(f'{dataset}/gt/{filename}.png')
            gt_masks.append(gt_mask)

        mask_iou_ = avg_mask_iou(pred_masks, gt_masks)
        mask_ious_diff_layers.append(mask_iou_)
        boundary_iou_ = avg_boundary_iou(pred_masks, gt_masks)
        boundary_ious_diff_layers.append(boundary_iou_)
        print(f"Token merging in layer {layer_i}: mask_iou: {mask_iou_}, boundary_iou: {boundary_iou_}")

    return mask_ious_diff_layers, boundary_ious_diff_layers


def get_args_parser():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of the segmentation result")

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset on which to run the SAM model')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the directory where masks and evaluation results will be stored')
    parser.add_argument('--model-type', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b',
                        help='The type of SAM model to load')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the SAM checkpoint to use for mask generation')
    parser.add_argument('--device', type=str, default="mps",
                        help='The device to run generation on.')
    parser.add_argument('--multiple-masks', action='store_true',
                        help='Enable multiple mask outputs. Require --num-masks to be specified.')
    parser.add_argument('--num-masks', type=int, required=True,
                        help='Specify the number of masks to output (only if --multiple-masks is set).')
    # TODO: mode and required parameters
    parser.add_argument('--tome-layers', type=str,)

    return parser.parse_args()


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
    if __name__ == "__main__":
        args = get_args_parser()
        evaluate()