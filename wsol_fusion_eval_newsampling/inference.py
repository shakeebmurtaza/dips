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

from ctypes import util
import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd
import utils
import torch

from .evaluation import BoxEvaluator
from .evaluation import MaskEvaluator
from .evaluation import configure_metadata
from .evaluation import get_mask
from .wsol_utils import t2n
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm

# _IMAGENET_MEAN = [0.485, .456, .406]
# _IMAGENET_STDDEV = [.229, .224, .225]
# _RESIZE_LENGTH = 224


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

def normalize_tensor(D):
    B = D.shape[0]
    C = D.shape[1]
    D_ = D.view(B,C,-1) 
    D_max = D_.max(dim=2)[0].unsqueeze(2).unsqueeze(2) 
    D_norm = (D/D_max).view(*D.shape)
    return D_norm

def re_normalize_cam(cam: torch.Tensor, h: float):
    _cam = cam + 1e-6
    e = torch.exp(_cam * h)
    e = e / e.max() # in [0, 1]
    e = torch.nan_to_num(e, nan=0.0, posinf=1., neginf=0.0)
    return e

class CAMComputer(object):
    def __init__(self, model, classifier, fusioscore_net, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None, use_image_as_input_for_fusion=False, args=None):
        self.multi_contour_eval = multi_contour_eval
        self.model = model
        self.fusioscore_net = fusioscore_net
        self.model.eval()
        self.fusioscore_net.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.classifier = classifier
        self.use_image_as_input_for_fusion = use_image_as_input_for_fusion
        self.args = args
        self.cam_save_dir = os.path.join(self.log_folder, 'output_samples')
        self.cam_save_dir_with_bbox = os.path.join(self.log_folder, 'output_samples_with_bbox')
        os.makedirs(self.cam_save_dir, exist_ok=True)
        os.makedirs(self.cam_save_dir_with_bbox, exist_ok=True)

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator,
                          "Ericsson": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval, num_heads=model.num_heads+1 if args.evaluate_only_on_attetion_maps else 1,
                                          _RESIZE_LENGTH=self.args.crop_size)#model.num_heads)

    def compute_and_evaluate_cams(self):
        with torch.no_grad():
            print("Computing and evaluating cams.")
            number_saved_samples = 0
            for images, imges_for_trans_fg, targets, image_ids in tqdm(self.loader):
                image_size = images.shape[2:]
                images = images.cuda()
                targets = targets.cuda()
                w_featmap = images.shape[-2] // self.model.patch_embed.patch_size
                h_featmap = images.shape[-1] // self.model.patch_embed.patch_size
                attentions = self.model.get_last_selfattention(images)
                nh = attentions.shape[1]  # number of head
                attentions = attentions[:, :, 0, 1:].reshape(-1, nh, w_featmap, h_featmap)  # attentions.reshape(-1, nh, w_featmap, h_featmap)
                attentions = torch.nan_to_num(attentions, nan=0.0, posinf=1., neginf=0.0)
                attentions = nn.functional.interpolate(attentions, size=image_size, mode="bilinear", align_corners=False)

                if self.args.evaluate_only_on_attetion_maps:
                    for ind, (attention, image_id) in enumerate(zip(normalize_tensor(attentions), image_ids)):
                        attention = torch.cat([torch.mean(attention, dim=0).unsqueeze(0), attention]).cpu().numpy()
                        self.evaluator.accumulate(attention.astype(np.float64), image_id)
                else:
                    attentions = normalize_tensor(attentions)[:, :self.args.number_of_heads_for_fusion, :, :]
                    if self.use_image_as_input_for_fusion:
                        attentions = torch.cat([attentions, images], axis=1)

                    if self.args.test_with_synth_inp:
                        new_attn = (attentions*0)
                        new_attn[:, :, 0:20, 0:20] = 0.5
                    else:
                        new_attn = attentions

                    cam_fused = self.fusioscore_net(new_attn.detach())#self.fusioscore_net(torch.cat([attentions.detach(), images.detach()], dim=1))#self.fusioscore_net(attentions.detach())
                    cam_fused = torch.softmax(cam_fused, dim=1)

                    if self.args.classifier_for_top1_top5 != None:
                        if self.args.classifier_for_top1_top5 == 'Dino_Head':
                            intermediate_output = self.model.get_intermediate_layers(images, self.args.n_last_blocks)
                            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                            if self.args.avgpool_patchtokens:
                                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                                output = output.reshape(output.shape[0], -1)
                            output_dict = self.classifier(output)
                        else:
                            output_dict = self.classifier(imges_for_trans_fg.cuda()) if self.args.classifier_for_top1_top5 == 'TransFg' else self.classifier(images)
                        pred_class = output_dict.softmax(dim=1)
                        _, pred_classes = pred_class.topk(5, 1, True, True)

                    for ind, (attention, image_id, img, target) in enumerate(zip(cam_fused, image_ids, images, targets)):
                        if self.args.classifier_for_top1_top5 != None:
                            pred_cls = pred_classes[ind]
                        selected_map_index = target+1 if self.args.seperate_map_for_each_class else 1
                        cams_normalized = attention.cpu().numpy().astype(np.float64)[selected_map_index:selected_map_index+1]
                        if self.args.classifier_for_top1_top5 != None:
                            best_bbox = self.evaluator.accumulate_with_all_matrices(cams_normalized, image_id, pred_cls, target)
                        else:
                            self.evaluator.accumulate(cams_normalized, image_id)
                        
        if self.args.classifier_for_top1_top5 == None:   
            matrices = self.evaluator.compute()
        else:
            matrices, localization_accuracies_at_each_th = self.evaluator.compute_on_all_matrcies()
            np.save(os.path.join(self.args.log_folder, f"{self.split}_localization_accuracies_at_each_th.npy"), localization_accuracies_at_each_th)
        return matrices