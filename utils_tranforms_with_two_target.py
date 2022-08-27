#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional



def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = functional.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, target2):
        for t in self.transforms:
            image, target, target2 = t(image, target, target2)
        return image, target, target2


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, target2):
        image = functional.resize(image, self.size)
        target = functional.resize(target, self.size)
        target2 = functional.resize(target2, self.size)
        return image, target, target2


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = functional.resize(image, size)
        target = functional.resize(target, size, interpolation=transforms.InterpolationMode.NEAREST)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, target2):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=0)
        target2 = pad_if_smaller(target2, self.size, fill=0)
        crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = functional.crop(image, *crop_params)
        target = functional.crop(target, *crop_params)
        target2 = functional.crop(target2, *crop_params)
        return image, target, target2


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target, target2):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
            target2 = functional.hflip(target2)
        return image, target, target2


class PILToTensor:
    def __call__(self, image, target):
        image = functional.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ToTensor:
    def __call__(self, image, target, target2):
        image = functional.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        target2 = torch.as_tensor(np.array(target2), dtype=torch.int64)
        return image, target, target2

class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = functional.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, target2):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image, target, target2


class ReduceLabels:
    def __call__(self, image, target):
        if not isinstance(target, np.ndarray):
            target = np.array(target).astype(np.uint8)
        # avoid using underflow conversion
        target[target == 0] = 255
        target = target - 1
        target[target == 254] = 255

        target = Image.fromarray(target)
        return image, target
