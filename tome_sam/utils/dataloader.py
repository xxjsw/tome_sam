# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import normalize
import torch.nn.functional as F

### main modification: one dataset instead of a list of dataset, with type hint, remove unnecessary
### dictionary storages, remove distributed settings


#### --------------------- dataloader online ---------------------####

@dataclass
class ReadDatasetInput:
    name: str
    im_dir: str
    gt_dir: Optional[str] # Ground truth directory may be empty
    im_ext: str
    gt_ext: str

@dataclass
class ReadDatasetOutput:
    name: str
    im_path: List[str]
    gt_path: List[str]
    im_ext: str
    gt_ext: str


def get_im_gt_name_dict(dataset: ReadDatasetInput, flag='valid') -> ReadDatasetOutput:
    print("------------------------------", flag, "--------------------------------")

    print("--->>>", "dataset: ", dataset.name, "<<<---")
    tmp_im_list = glob(dataset.im_dir + os.sep + '*' + dataset.im_ext)
    print('-im-', dataset.name, dataset.im_dir, ': ', len(tmp_im_list))

    if dataset.gt_dir == "":
        print('-gt-', dataset.name, dataset.gt_dir, ': ', 'No Ground Truth Found')
        tmp_gt_list = []
    else:
        tmp_gt_list = [
            dataset.gt_dir + os.sep + x.split(os.sep)[-1].split(dataset.im_ext)[0] +
            dataset.gt_ext for x in tmp_im_list]
        print('-gt-', dataset.name, dataset.gt_dir, ': ', len(tmp_gt_list))

    return ReadDatasetOutput(name=dataset.name,
                             im_path=tmp_im_list,
                             gt_path=tmp_gt_list,
                             im_ext=dataset.im_ext,
                             gt_ext=dataset.gt_ext)

def create_dataloaders(name_im_gt_path: ReadDatasetOutput,
                       my_transforms: List= None,
                       batch_size=1,
                       training=False) -> tuple[DataLoader, Dataset]:

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8

    if training:
        gos_dataset = OnlineDataset(name_im_gt_path, transform=transforms.Compose(my_transforms))
        sampler = RandomSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        gos_dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train,
                                    num_workers=num_workers_)

    else:
        gos_dataset = OnlineDataset(name_im_gt_path, transform=transforms.Compose(my_transforms),
                                    eval_ori_resolution=True)
        sampler = RandomSampler(gos_dataset)
        gos_dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False,
                                    num_workers=num_workers_)

    return gos_dataloader, gos_dataset


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), self.size, mode='bilinear'), dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), self.size, mode='bilinear'), dim=0)
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size)}


class RandomCrop(object):
    def __init__(self, size=[288, 288]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top:top + new_h, left:left + new_w]
        label = label[:, top:top + new_h, left:left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image, self.mean, self.std)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), scaled_size.tolist(), mode='bilinear'),
                                     dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), scaled_size.tolist(), mode='bilinear'),
                                     dim=0)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
        label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(image.shape[-2:])}


class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_path: ReadDatasetOutput, transform=None, eval_ori_resolution=False):

        # print("name_im_gt_path", name_im_gt_path)
        self.transform = transform
        self.dataset = {}

        im_name_list = []
        im_path_list = []
        gt_path_list = []
        im_ext_list = []
        gt_ext_list = []

        im_name_list.extend(
                [x.split(os.sep)[-1].split(name_im_gt_path.im_ext)[0] for x in name_im_gt_path.im_path])
        im_path_list.extend(name_im_gt_path.im_path)
        gt_path_list.extend(name_im_gt_path.gt_path)
        im_ext_list.extend([name_im_gt_path.im_ext for x in name_im_gt_path.im_path])
        gt_ext_list.extend([name_im_gt_path.gt_ext for x in name_im_gt_path.gt_path])


        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "shape": torch.tensor(im.shape[-2:]),
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample