import random
from dataclasses import dataclass
from typing import Optional, List

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt, patches
from skimage import io, measure

from tome_sam.build_tome_sam import tome_sam_model_registry
from tome_sam.utils import misc
from tome_sam.utils.tome_presets import SAMToMeSetting, ToMeConfig, PiToMe, ToMe, ToMeSD
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


@dataclass
class VisualizeArgs:
    input_image: str
    input_mask: str
    output: str
    model_type: str
    checkpoint: str
    seed: int
    input_size: List[int]
    tome_setting: Optional[SAMToMeSetting] = None

def plot_image_mask_bbox(image, pred_mask, gt_mask, bounding_box, save_path='output.png'):
    """
    Visualize an image with an overlayed mask and bounding box
    Args:
        image(torch.Tensor): (3, H, W)
        pred_mask(torch.Tensor): (1, H, W), with boolean values
        gt_mask(torch.Tensor): (1, H, W), with boolean values
        bounding_box(torch.Tensor): (1, 4), [x_min, y_min, x_max, y_max]
        save_path: path to save the output image
    """
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    pred_mask = pred_mask.squeeze(0).numpy()
    gt_mask = gt_mask.squeeze(0).numpy()
    bbox = bounding_box.squeeze(0).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    # overlay masks
    ax.imshow(image)
    ax.imshow(pred_mask, cmap='Reds', alpha=0.2)
    ax.imshow(gt_mask, cmap='Greens', alpha=0.2)

    pred_contours = measure.find_contours(pred_mask, 0.5)
    gt_contours = measure.find_contours(gt_mask, 0.5)

    # draw contours
    for contour in pred_contours:
        ax.plot(contour[:, 1], contour[:, 0], 'red', linewidth=1, label='pred_mask')

    for contour in gt_contours:
        ax.plot(contour[:, 1], contour[:, 0], 'darkgreen', linewidth=1, label='gt_mask')

    # bounding boxes
    x_min, y_min, x_max, y_max = bbox
    bbox_rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(bbox_rect)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Segmentation output image saved to {save_path}')



def visualize_output_mask(args: VisualizeArgs):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tome_sam = tome_sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        tome_setting=args.tome_setting,
    )

    tome_sam.eval()

    im = io.imread(args.input_image)
    gt = io.imread(args.input_mask)
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im = torch.tensor(im.copy(), dtype=torch.float32)
    im = torch.transpose(torch.transpose(im, 1, 2), 0, 1) # (3, H, W)
    gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0) # (1, H, W)

    # Resize
    image = torch.squeeze(F.interpolate(torch.unsqueeze(im, 0), args.input_size, mode='bilinear'), dim=0)
    gt_mask = torch.squeeze(F.interpolate(torch.unsqueeze(gt, 0), args.input_size, mode='bilinear'), dim=0) # (1, H, W)

    bounding_box = misc.masks_to_boxes(gt_mask[0].unsqueeze(0)) # (1, 4)

    dict_input = dict()
    dict_input['image'] = image.to(torch.uint8)
    dict_input['boxes'] = bounding_box
    dict_input['original_size'] = image.shape[1:]

    with torch.no_grad():
        mask = tome_sam([dict_input], multimask_output=False)[0]['masks'][0] # (1, H, W)

    m_iou = misc.mask_iou(mask, gt_mask)
    b_iou = misc.boundary_iou(gt_mask, mask)
    print(f'Mask IoU: {m_iou}, Boundary IoU: {b_iou}')
    plot_image_mask_bbox(image, mask, gt_mask, bounding_box, save_path=args.output)



if __name__ == '__main__':
    tomesd_setting: SAMToMeSetting = {
        7: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=False)
        ),
        8: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=False)
        ),
        9: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=False)
        ),
        10: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=False)
        ),
        11: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=False)
        ),
    }

    tome_setting: SAMToMeSetting = {
        7: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
        8: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
        9: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
        10: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
        11: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
    }

    tome25_setting: SAMToMeSetting = {
        7: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
        8: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
        9: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
        10: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
        11: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
    }

    pitome_setting: SAMToMeSetting = {
        7: ToMeConfig(
            mode='pitome',
            params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
        ),
        8: ToMeConfig(
            mode='pitome',
            params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
        ),
        9: ToMeConfig(
            mode='pitome',
            params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
        ),
        10: ToMeConfig(
            mode='pitome',
            params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
        ),
        11: ToMeConfig(
            mode='pitome',
            params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
        ),
    }

    args = VisualizeArgs(
        input_image='data/DIS5K/DIS-VD/im/11#Furniture#17#Table#49706461457_de5227b966_o.jpg',
        input_mask='data/DIS5K/DIS-VD/gt/11#Furniture#17#Table#49706461457_de5227b966_o.png',
        output='output_pitome.png',
        model_type="vit_b",
        checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        seed=42,
        input_size=[1024, 1024],
        tome_setting=pitome_setting,
    )

    visualize_output_mask(args)