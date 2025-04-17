import random
from dataclasses import dataclass
from typing import Optional, List

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt, patches
from skimage import io, measure

from segment_anything import SamAutomaticMaskGenerator
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


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def visualize_automatic_mask_generator(args: VisualizeArgs, original_resolution=False):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tome_sam = tome_sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        tome_setting=args.tome_setting,
    )

    mask_generator = SamAutomaticMaskGenerator(model=tome_sam,
                                               points_per_side=32,
                                                pred_iou_thresh=0.86,
                                                stability_score_thresh=0.92,
                                                crop_n_layers=1,
                                                crop_n_points_downscale_factor=2,
                                                min_mask_region_area=100)
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    plt.close(fig)




def visualize_output_mask(args: VisualizeArgs, original_resolution=False):
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
    _, original_H, original_W = im.shape
    gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0) # (1, H, W)

    # Resize
    resized_im = torch.squeeze(F.interpolate(torch.unsqueeze(im, 0), args.input_size, mode='bilinear'), dim=0)
    resized_gt= torch.squeeze(F.interpolate(torch.unsqueeze(gt, 0), args.input_size, mode='bilinear'), dim=0) # (1, H, W)

    resized_bounding_box = misc.masks_to_boxes(resized_gt[0].unsqueeze(0)) # (1, 4)

    dict_input = dict()
    dict_input['image'] = resized_im.to(torch.uint8)
    dict_input['boxes'] = resized_bounding_box
    dict_input['original_size'] = resized_im.shape[1:]

    with torch.no_grad():
        resized_mask = tome_sam([dict_input], multimask_output=False)[0]['masks'][0] # (1, H, W)

    # evaluation on original resolution
    mask = torch.squeeze(F.interpolate(torch.unsqueeze(resized_mask.float(), 0), [original_H, original_W], mode='bilinear'), dim=0)
    bounding_box = misc.masks_to_boxes(gt[0].unsqueeze(0))
    m_iou = misc.mask_iou(mask, gt)
    b_iou = misc.boundary_iou(gt, mask)
    print(f'Mask IoU: {m_iou}, Boundary IoU: {b_iou}')
    if original_resolution:
        plot_image_mask_bbox(im, mask, gt, bounding_box, save_path=args.output)
    else:
        plot_image_mask_bbox(resized_im, resized_mask, resized_gt, resized_bounding_box, save_path=args.output)


if __name__ == '__main__':
    tomesd_setting: SAMToMeSetting = {
        2: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
        ),
        5: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
        ),
        8: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
        ),
        11: ToMeConfig(
            mode='tomesd',
            params=ToMeSD(r=0.5, sx=2, sy=2, no_rand=True)
        ),
    }

    tome_setting: SAMToMeSetting = {
        2: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
        5: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
        8: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
        11: ToMeConfig(
            mode='tome',
            params=ToMe(r=0.5)
        ),
    }

    grad_tome_setting: SAMToMeSetting = {
        11: ToMeConfig(
            mode='grad_tome',
            params=ToMe(r=0.7)
        ),
    }

    tome25_setting: SAMToMeSetting = {
        2: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
        5: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
        8: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
        11: ToMeConfig(
            mode='tome25',
            params=ToMe(r=0.5)
        ),
    }

    pitome_setting: SAMToMeSetting = {
        11: ToMeConfig(
            mode='pitome',
            params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
        ),
    }

    args = VisualizeArgs(
        input_image='data/DIS5K/DIS-VD/im/11#Furniture#17#Table#49706461457_de5227b966_o.jpg',
        input_mask='data/DIS5K/DIS-VD/gt/11#Furniture#17#Table#49706461457_de5227b966_o.png',
        output='automatic_mask_generator.png',
        model_type="vit_b",
        checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        seed=42,
        input_size=[1024, 1024],
        tome_setting=None,
    )

    visualize_output_mask(args)
    # visualize_automatic_mask_generator(args)