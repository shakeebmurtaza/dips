"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch
import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd

from .evaluation import BoxEvaluator
from .evaluation import MaskEvaluator as MaskEvaluator
from .evaluation import configure_metadata
from .wsol_utils import t2n
import torch.nn as nn

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval, num_heads=model.num_heads)

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            #attention to cam
            # _featmap = images.shape[-2] // self.model.patch_embed.patch_size
            w_featmap = images.shape[-2] // self.model.patch_embed.patch_size
            h_featmap = images.shape[-1] // self.model.patch_embed.patch_size
            img_size = images.shape[-2]
            attentions = self.model.get_last_selfattention(images)
            nh = attentions.shape[1]  # number of head
            # attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            # attentions = attentions.reshape(nh, w_featmap, h_featmap)
            # # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=img_size, mode="bicubic")[0].cpu().numpy()
            # cams = t2n(attentions)

            #attention to cam
            # cams = t2n(self.model.get_last_selfattention(images))
            for attention, image_id in zip(attentions, image_ids):
                attention = attention[:, 0, 1:].reshape(nh, -1)
                attention = attention.reshape(nh, w_featmap, h_featmap)
                cams = t2n(attention)
                cams_normalized = []
                for cam in cams:
                    cam_resized = cv2.resize(cam, image_size, interpolation=cv2.INTER_CUBIC)
                    cams_normalized.append(normalize_scoremap(cam_resized))
                cams_normalized = np.asarray(cams_normalized)
                # if self.split in ('val', 'test'):
                #     cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                #     if not os.path.exists(ospd(cam_path)):
                #         os.makedirs(ospd(cam_path))
                #     np.save(ospj(cam_path), cams_normalized)
                self.evaluator.accumulate(cams_normalized, image_id)
        torch.cuda.synchronize()
        return self.evaluator.compute()
