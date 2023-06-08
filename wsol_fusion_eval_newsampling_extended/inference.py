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
import copy

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
            # acc_ = []
            # acc_indv = {0: [], 1: [],2: [], 3: [],4: [], 5: []}
            # img_acc_ = []
            # acc_count = 0
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
                orig_attentions = copy.deepcopy(attentions)

                if self.args.evaluate_only_on_attetion_maps:
                    for ind, (attention, image_id) in enumerate(zip(normalize_tensor(attentions), image_ids)):
                        attention = torch.cat([torch.mean(attention, dim=0).unsqueeze(0), attention]).cpu().numpy()
                        self.evaluator.accumulate(attention.astype(np.float64), image_id)
                else:
                    # attentions = normalize_tensor(attentions)[:, :self.args.number_of_heads_for_fusion, :, :]
                    attentions = attentions[:, :self.args.number_of_heads_for_fusion, :, :]
                    if self.use_image_as_input_for_fusion:
                        attentions = torch.cat([attentions, images], axis=1)

                    if self.args.test_with_synth_inp:
                        new_attn = (attentions*0)
                        new_attn[:, :, 0:20, 0:20] = 0.5
                    else:
                        new_attn = attentions

                    # cam_fused, class_logits = self.fusioscore_net(new_attn.detach())#self.fusioscore_net(torch.cat([attentions.detach(), images.detach()], dim=1))#self.fusioscore_net(attentions.detach())
                    
        
                    ############################Get CAMs############################
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        x_out, attentions = self.model.forward_get_all_heads(images)
                        x_cls, x_patch = x_out[:, 0], x_out[:, 1:]
                        n, p, c = x_patch.shape
                        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
                        x_patch = x_patch.permute([0, 3, 1, 2])
                        x_patch = x_patch.contiguous()
                    cam_fused = self.fusioscore_net(x_patch)
                    # print('wait')
                    ############################Get CAMs############################
                    
                    cam_fused = torch.softmax(cam_fused, dim=1)

                   
                    if self.args.classifier_for_top1_top5 != None:
                        if not self.args.use_dino_classifier:
                            logits = self.classifier(images)
                            if self.args.dataset_name == 'OpenImages':
                                class_logits = logits['logits']
                        else:
                            intermediate_output = self.model.get_intermediate_layers(images, self.args.n_last_blocks)
                            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                            if self.args.avgpool_patchtokens:
                                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                                output = output.reshape(output.shape[0], -1)
                            class_logits = self.classifier(output)
                        
                        _, pred_classes = class_logits.topk(5, 1, True, True)

                    for ind, (attention, image_id, img, target, _orig_attentions) in enumerate(zip(cam_fused, image_ids, images, targets, orig_attentions)):
                        if self.args.classifier_for_top1_top5 != None:
                            pred_cls = pred_classes[ind]
                        selected_map_index = target+1 if self.args.seperate_map_for_each_class else 1
                        cams_normalized = utils.normalize_img_0_255_float(attention.cpu().numpy().astype(np.float64)[selected_map_index:selected_map_index+1])
                        
                        if self.args.run_accumulate_and_save_fun and self.args.dataset_name != 'OpenImages':
                            self.evaluator.accumulate_and_save(cams_normalized, image_id, img, self.cam_save_dir_with_bbox, dino_attention=_orig_attentions.cpu().numpy())
                        elif self.args.classifier_for_top1_top5 != None and self.args.dataset_name != 'OpenImages':
                            best_bbox = self.evaluator.accumulate_with_all_matrices(cams_normalized, image_id, pred_cls, target)
                        else:
                            self.evaluator.accumulate(cams_normalized, image_id)
                        if number_saved_samples < self.args.num_samples_to_save and (not self.args.run_accumulate_and_save_fun or self.args.dataset_name == 'OpenImages'):#and self.split == 'test:
                            img = utils.normalize_img_0_255(img.permute(1,2,0).cpu().numpy())
                            heatmap = cv2.applyColorMap((cams_normalized[0]*255).astype(np.uint8), cv2.COLORMAP_JET)
                            heatmap = heatmap[..., ::-1].copy()
                            dest = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0.0)
                            if self.args.dataset_name == 'OpenImages':
                                gt_mask = get_mask(self.evaluator.mask_root,
                                    self.evaluator.mask_paths[image_id],
                                    self.evaluator.ignore_paths[image_id],
                                    self.evaluator.resize_length)
                                
                                fig, axs = plt.subplots(1, 4)
                                axs[0].axis('off')
                                axs[0].set_title('Target')
                                axs[0].imshow(gt_mask, cmap='gray')
                                axs[1].axis('off')
                                axs[1].imshow(dest, cmap='gray')
                                axs[1].set_title('Predicted')
                                axs[2].axis('off')
                                axs[2].imshow(img, cmap='gray')
                                axs[2].set_title('Image')

                                axs[3].set_ylim([0, 105])
                                nb_bins = 10000
                                bins= [i for i in range(nb_bins)]
                                hist = np.histogram(cams_normalized.squeeze().reshape(-1), bins=nb_bins, range=[0, 1])[0]
                                axs[3].step(bins, hist, label = 'Our Method')
                                axs[3].set_title('Our Method'.upper(), fontsize=16)

                                plt.tight_layout()
                                pxap_score_to_save = self.evaluator.PxAP_per_image[0][image_id]
                                plt.savefig(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_')), dpi=300)
                                plt.close()
                                plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'gt_mask.png', gt_mask, dpi=300)
                                plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'gt_mask_gray.png', gt_mask, dpi=300, cmap='gray')
                                plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'img.png', img, dpi=300)
                                plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'our_map.png', cams_normalized[0], dpi=300)
                                plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'our_map_gray.png', cams_normalized[0], dpi=300, cmap='gray')
                                plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'predicted.png', dest, dpi=300, cmap='gray')
                                # plt.imsave(os.path.join(self.cam_save_dir, str(round(self.evaluator.PxAP_per_image[0][image_id], 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'g_map.png', (cams_normalized[0]*255).astype(np.uint8), dpi=300)
                                # plt.imsave(os.path.join(self.cam_save_dir, str(round(self.evaluator.PxAP_per_image[0][image_id], 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'g_map_nor_g.png', (utils.normalize_scoremap(cams_normalized[0])*255).astype(np.uint8), dpi=300, cmap='gray')
                                # plt.imsave(os.path.join(self.cam_save_dir, str(round(self.evaluator.PxAP_per_image[0][image_id], 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'g_map_g.png', (cams_normalized[0]*255).astype(np.uint8), dpi=300, cmap='gray')
                                for ind__orig_attention, _orig_attention in enumerate(_orig_attentions):
                                    plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'dino_attn'+str(ind__orig_attention)+'.png', (utils.normalize_scoremap(_orig_attention.cpu().numpy())*255).astype(np.uint8), dpi=300)
                                    plt.imsave(os.path.join(self.cam_save_dir, str(round(pxap_score_to_save, 4)).replace('.', '_') + ' - ' +image_id.replace('/', '_') )+ 'dino_attn'+str(ind__orig_attention)+'_gray.png', (utils.normalize_scoremap(_orig_attention.cpu().numpy())*255).astype(np.uint8), dpi=300, cmap='gray')

                            else:
                                plt.imsave(os.path.join(self.cam_save_dir, image_id.replace('/', '_')), dest)
                            number_saved_samples += 1
                    
        if self.args.classifier_for_top1_top5 == None:   
            matrices = self.evaluator.compute()
        else:
            matrices, localization_accuracies_at_each_th = self.evaluator.compute_on_all_matrcies()
            np.save(os.path.join(self.args.log_folder, f"{self.split}_localization_accuracies_at_each_th.npy"), localization_accuracies_at_each_th)
        # if self.split == 'test' and self.multi_contour_eval == False:
        #     for i in acc_indv.keys():
        #         acc_indv[i] = np.mean(acc_indv[i])
        #     matrices['acc_cam_ind'] = acc_indv
        return matrices