# Copyright (c) Facebook, Inc. and its affiliates.
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
# limitations under the License.
# import wandb
# wandb.init(project="train_fusion_module_soufiane", entity="shakeebmurtaza")
from tqdm import tqdm
from einops import rearrange

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from torch.autograd import Variable
import pickle
from comet_ml import Optimizer, Experiment
import comet_ml
import uuid

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision import models
# from matplotlib import pyplot as plt
# from munch import DefaultMunch
# import optuna
from wsol.resnet import resnet50
import torch.optim.lr_scheduler as lrs
from wsol_eval_extract_bbox.wsol.resnet import resnet50 as resnet50_wsol_eval

import utils
import vision_transformer as vits
# from vision_transformer import DINOHead
from wsol_fusion_eval_newsampling.data_loaders import get_data_loader_non_dist as get_data_loader
from wsol_fusion_eval_newsampling.data_loaders_overlapped import get_data_loader_non_dist as get_data_loader_overlapped
from wsol.wsol_utils import str2bool, configure_data_paths, configure_log_folder

from wsol_fusion_eval_newsampling.inference import CAMComputer
# from wsol.inference import CAMComputer as CAMComputer_for_six_heads
# from sam_optimizer.sam import SAM
import copy
# from efficientnet_pytorch import EfficientNet
# from filelock import FileLock
from dlib.unet.model import Unet
from aux.self_learning import MBSeederSLCAMS, MBProbSeederSLCAMS, MBProbSeederSLCamsWithROI
from aux import self_learning
from aux.crf.dense_crf_loss import ConRanFieldFcams
from wsol_fusion_eval_newsampling.inference import normalize_tensor
# from aux.self_learning import BinaryRegionsProposal

# from constraints import WeightsConstraints

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('FusionModule', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],# \
                # + torchvision_archs + torch.hub.list("facebookresearch/xcit"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    # parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
    #     to use half precision for training. Improves training time and memory requirements,
    #     but can provoke instability and slight decay of performance. We recommend disabling
    #     mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=0, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    # parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
    #     end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='sgd', type=str,
        choices=['adamw', 'sgd', 'lars', 'sam'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    #Evaluation Paramters
    parser.add_argument('--metadata_root', type=str, default='metadata/')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')
    _DATASET_NAMES = ('CUB', 'ILSVRC', 'OpenImages', 'Ericsson')
    parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='dataset/',
                        help='path to dataset images')
    #Eval WSOL Paramters
    parser.add_argument('--cam_curve_interval', type=float, default=0.05,
                        help="At which threshold intervals will the score maps "
                             "be evaluated?.")
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])
    parser.add_argument('--mask_root', type=str, default='dataset/',
                        help="Root folder of masks (OpenImages).")
    parser.add_argument('--experiment_name', type=str, default='dino_testing')
    parser.add_argument('--override_cache', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--dino_pretrained', type=str, default='',
                        help="pretrained dino.")
    parser.add_argument('--resnet_pretrained', type=str, default='pre-trained/',
                        help="pretrained dino on (OpenImages).")

    parser.add_argument('--wd_lmbda', type=float, default=0.3)
    parser.add_argument('--lmda_crf_fc', type=float, default=2e-09)
    parser.add_argument('--experiment_category', type=str, default='train_log/',
                        help="Root folder for saving experiemnt detials.")

    #seed method parameters
    parser.add_argument('--seed_type', default='MBProbSeederSLCAMS', type=str,
        choices=self_learning.__all__, help="""Seed Selection Type""")
    parser.add_argument('--seed_min', type=int, default=1,
                        help="number of pixels to sample from background")
    parser.add_argument('--seed_max', type=int, default=1,
                        help="number of pixels to sample from foreground")
    parser.add_argument('--seed_min_p', type=utils.restricted_float, default=0.3,
                        help="[0, 1]. percentage (from entire image) of pixels to be considered background. min_ pixels will be sampled from these pixels. IMPORTANT.")
    parser.add_argument('--seed_fg_erode_k', type=int, default=11,
                        help="erosion kernel of the cam before sampling. 11 is fine.")
    parser.add_argument('--seed_fg_erode_iter', type=int, default=1,
                        help="number iteration to perform the erosion. 1 is fine.")
    parser.add_argument('--seed_ksz', type=int, default=3,
                        help="int. kernel size to dilate the seeds.")
    parser.add_argument('--seed_seg_ignore_idx', type=int, default=-255,
                        help="param seed_seg_ignore_idx: int. index for unknown seeds. 0: background")

    parser.add_argument('--lmda_clsloss', type=float, default=1.0,
                        help="Weight of class loss")
    parser.add_argument('--lmda_seed_loss', type=float, default=1.0,
                        help="Weight of seed loss")
    
    parser.add_argument('--search_hparameters', type=str2bool, nargs='?',
                        const=True, default=False)

    parser.add_argument('--run_for_best_setting', type=str2bool, nargs='?',
                        const=True, default=False)

    parser.add_argument('--only_create_study', type=str2bool, nargs='?',
        const=True, default=False)

    parser.add_argument('--use_image_as_input_for_fusion', type=str2bool, nargs='?',
                        const=True, default=False)

    parser.add_argument('--number_of_heads_for_fusion', type=int, nargs='?', default=6)

    parser.add_argument('--evaluation_type', default='all_heads', type=str,
        choices=['all_heads_and_mean', '1_5_heads_and_mean', '1_4_heads_and_mean', 
        '1_5_heads_and_mean_1_5', '1_4_heads_and_mean_1_4', 'all_heads_and_mean_1_4'])

    parser.add_argument('--iou_regions_path', required=True, type=str)

    parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')

    parser.add_argument('--lr_decay', type=int, default=3,
                    help='learning rate decay per N epochs')

    parser.add_argument('--best_hp_index_for_full_exp', type=int, default=0,
                    help='0 for the top 1 and so on')

    parser.add_argument('--exp_pre_name', type=str, default='')

    parser.add_argument('--test_with_synth_inp', default=False, type=str2bool)

    parser.add_argument('--normalize_with_softmax', default=False, type=str2bool)

    parser.add_argument('--save_checkpoint_each_epoch', default=False, type=str2bool)

    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")

    parser.add_argument('--use_dino_classifier', default=False, type=utils.bool_flag)
    parser.add_argument('--use_classifier', default=False, type=utils.bool_flag)

    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on test set')
    parser.add_argument('--run_for_given_setting', dest='run_for_given_setting', action='store_true', help='run full exp for given settings')
    parser.add_argument('--evaluate_checkpoint', type=str, default='', help="checkpoint used for evaluation on test set")
    parser.add_argument('--run_accumulate_and_save_fun', action='store_true')
    # parser.add_argument('--eval_with_all_matrices', action='store_true')
    parser.add_argument('--crf_fc', action='store_true', help='to turn on crf_fc loss')
    parser.add_argument('--seed_tech', type=str, default=self_learning._WEIGHTED,
                        choices=(self_learning._WEIGHTED, self_learning._UNIFORM))
    parser.add_argument('--seed_tech_bg', type=str, default=self_learning._UNIFORM,
                        choices=(self_learning._WEIGHTED, self_learning._UNIFORM))
    parser.add_argument('--classifier_for_top1_top5', type=str, default=None,
                        choices=('TransFg', 'Dino_Head'))
    parser.add_argument('--seperate_map_for_each_class', action='store_true', help='use seperate map for each class')
    
    parser.add_argument('--evaluate_only_on_attetion_maps', action='store_true', help='use seperate map for each class')

    parser.add_argument('--hp_search_type', default='bayes', type=str, choices=['grid', 'bayes'])

    parser.add_argument('--use_full_bg_with_no_fg_regions', action='store_true', help='For ericsson dataset use full bg regions for loss inseted of just some pixels that contain no foregournd regions')

    parser.add_argument('--unet_encoder_name', default='resnet50', type=str, choices=['resnet50', 'vgg16', 'inceptionv3']) 
    
    parser.add_argument('--is_certain_to_uncertain_train', action='store_true', help='Certain to uncertain train is true or not')
    parser.add_argument('--is_certain_to_uncertain_train_iou_regions_root', default=None, help='Root dir for iou regions path')
    parser.add_argument('--certain_to_uncertain_th', type=float, nargs='+',
                        default=[.8, .75, .7, .65, .6, .55, .5, .45, .4, .35])

    parser.add_argument('--top_n_head_for_sampling', type=int, default=-1, help='use top n heads for sampling if greater than 0')

    parser.add_argument('--num_samples_to_save', type=int, default=0)
    parser.add_argument('--discrete_search_space', action='store_true', help='use discrete search space for hp search')
    
    # parser.add_argument('--import_from_optuna', type=str2bool, nargs='?', const=True, default=False)

    return parser

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")

def _compute_accuracy(classifier, loader, teacher, use_dino_classifier, n_last_blocks, args):
    num_correct = 0
    num_images = 0

    for i, (images, imges_for_trans_fg, targets, image_ids) in enumerate(loader):
        images = imges_for_trans_fg if args.classifier_for_top1_top5 == 'TransFg' else images
        images = images.cuda()
        targets = targets.cuda()
        if use_dino_classifier:
            intermediate_output = teacher.get_intermediate_layers(images, n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if args.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
            output_dict = classifier(output)
        else:
            output_dict = classifier(images)
        pred = output_dict.argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

    classification_acc = num_correct / float(num_images) * 100
    return classification_acc

def normalize_scoremap(attention, scores):
    for i in range(len(attention)):
        for j in range(len(attention[0])):
            attention[i][j] = ((attention[i][j] - attention[i][j].min()) / (attention[i][j].max() - attention[i][j].min()))*scores[i][j]
    return attention


def evaluate_fusionmodule(epoch, split, model, classifier, loaders, fusioscorenet, args, evaluate_on_MaxBoxV2 = False):
    print("Evaluate epoch {}, split {}".format(epoch, split))
    model.eval()
    fusioscorenet.eval()
    if classifier != None:
        classifier.eval()
    loc_score = None
    
    cam_computer = CAMComputer(
        model=model,
        classifier=classifier,
        fusioscore_net=fusioscorenet,
        loader=loaders[split],
        metadata_root=os.path.join(args.metadata_root, split),
        mask_root=args.mask_root,
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=args.dataset_name,
        split=split,
        cam_curve_interval=args.cam_curve_interval,#args.cam_curve_interval,
        multi_contour_eval=False,#args.multi_contour_eval,
        log_folder=args.log_folder,
        use_image_as_input_for_fusion = args.use_image_as_input_for_fusion,
        args = args
    )
    cam_performance = cam_computer.compute_and_evaluate_cams()

    cam_performance['epoch'] = epoch
    with (Path(args.log_folder) / f"{split}_log.txt").open("a") as f:
        f.write(json.dumps(cam_performance) + "\n")
    
    if args.classifier_for_top1_top5 != None:
        with (Path(args.log_folder) / f"{split}_maxbox_top1_top5_loc.txt").open("a") as f:
            f.write(json.dumps({'top1': cam_computer.evaluator.top1, 'top5': cam_computer.evaluator.top5}) + "\n")

        with (Path(args.log_folder) / f"{split}_error.txt").open("a") as f:
            f.write(json.dumps({'cls_wrong': np.mean(cam_computer.evaluator.cls_wrong_top1_loc_cls), 
            'multi_instances': np.mean(cam_computer.evaluator.cls_wrong_top1_loc_mins),
            'region_part': np.mean(cam_computer.evaluator.cls_wrong_top1_loc_part),
            'region_more': np.mean(cam_computer.evaluator.cls_wrong_top1_loc_more),
            'region_wrong': np.mean(cam_computer.evaluator.cls_wrong_top1_loc_wrong)}) + "\n")

    if evaluate_on_MaxBoxV2:
        cam_computer = CAMComputer(
            model=model,
            classifier=classifier,
            fusioscore_net=fusioscorenet,
            loader=loaders[split],
            metadata_root=os.path.join(args.metadata_root, split),
            mask_root=args.mask_root,
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=args.dataset_name,
            split=split,
            cam_curve_interval=args.cam_curve_interval,#args.cam_curve_interval,
            multi_contour_eval=True,#args.multi_contour_eval,
            log_folder=args.log_folder,
            use_image_as_input_for_fusion = args.use_image_as_input_for_fusion,
            args = args
        )
        cam_performanceV2 = cam_computer.compute_and_evaluate_cams()
        loc_score = {}
        for key in cam_performanceV2.keys():
            try:
                loc_score[key] = np.average(cam_performanceV2[key])
            except:
                break

        # if split == 'test':
        loc_score['epoch'] = epoch
        with (Path(args.log_folder) / f"{split}_V2_log.txt").open("a") as f:
            f.write(json.dumps(loc_score) + "\n")

        cam_performanceV2['epoch'] = epoch
        with (Path(args.log_folder) / f"{split}_V2_log_without_avg.txt").open("a") as f:
            f.write(json.dumps(cam_performanceV2) + "\n")

        if args.classifier_for_top1_top5:
            with (Path(args.log_folder) / f"{split}_maxboxv2_top1_top5_loc.txt").open("a") as f:
                f.write(json.dumps({'top1': cam_computer.evaluator.top1, 'top5': cam_computer.evaluator.top5}) + "\n")

    if args.dataset_name == 'OpenImages':
        return cam_performance[1]
    return cam_performance[1][1]

def find_empty_rois(loader):
    empty_rois_lst = []
    empty_rois_lst_without_aug = []
    for _, _, img_id, roi, _, box_fused_input_without_aug in tqdm(loader):
        if roi.max().item() == 0:
            empty_rois_lst.append(img_id)
        if box_fused_input_without_aug.max().item() == 0:
            empty_rois_lst_without_aug.append(img_id)
        
    return empty_rois_lst, empty_rois_lst_without_aug

def train_fusionmodule(args, trial):
    cudnn.benchmark = True

    loaders = get_data_loader(
        data_roots=args.data_paths,
        metadata_root=args.metadata_root,
        batch_size=args.batch_size_per_gpu,
        workers=args.num_workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        proxy_training_set=args.proxy_training_set,
        num_val_sample_per_class=args.num_val_sample_per_class, args=args)

    data_loader = loaders['train_fusion']

   

    args.arch = args.arch.replace("deit", "vit")

    if args.arch in vits.__dict__.keys():
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)

    utils.load_pretrained_weights(teacher, args.dino_pretrained, 'teacher', args.arch, args.patch_size)
    

    teacher = teacher.cuda()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if args.dataset_name == 'Ericsson':
        num_labels = 2
    elif args.dataset_name == 'CUB':
        num_labels = 200
    elif args.dataset_name == 'ILSVRC':
        num_labels = 1000
    elif args.dataset_name == 'OpenImages':
        num_labels = 100
    else:
        raise ValueError('Invalid Datasetname')
    args.num_labels = num_labels

    fusion_channels = args.number_of_heads_for_fusion+3 if args.use_image_as_input_for_fusion else args.number_of_heads_for_fusion
    fusion_module_out_channels = num_labels+1 if args.seperate_map_for_each_class else 2
    fusioscore_net = Unet(encoder_name=args.unet_encoder_name, in_channels=fusion_channels, classes=fusion_module_out_channels).cuda()#FusionScores().cuda()

    class_loss = nn.CrossEntropyLoss()
    seed_loss = nn.CrossEntropyLoss(ignore_index=args.seed_seg_ignore_idx)

    crf_loss = ConRanFieldFcams(weight=args.lmda_crf_fc, sigma_rgb=15.0, sigma_xy=100.0, scale_factor=1.0, device='cuda')#nn.CrossEntropyLoss(ignore_index=args.seed_seg_ignore_idx)

    
    if args.use_classifier or args.classifier_for_top1_top5 != None:
        if (not args.use_dino_classifier) and (not args.evaluate):
            classifier = None
            if args.dataset_name == 'OpenImages':
                classifier = resnet50_wsol_eval('cam', num_classes=args.num_labels, large_feature_map=False, pretrained=False)#models.resnet50(pretrained=True).cuda()
                classifier_weight_path = 'wsol_eval_extract_bbox/wsol/last_checkpoint_resnet_openimages.pth.tar'
                state_dict = torch.load(classifier_weight_path)
                classifier.load_state_dict(state_dict['state_dict'])
                print("classifier's weights loaded from {}".format(classifier_weight_path))
            elif args.dataset_name == "ILSVRC":
                classifier = models.resnet50(pretrained=True)
            elif args.dataset_name == "Ericsson" or args.dataset_name == "CUB":
                pass
            else:
                raise ValueError("Invalid Value selected for dataset name")
        elif (args.use_dino_classifier and not args.evaluate) or (args.evaluate and args.classifier_for_top1_top5 == 'Dino_Head'):
            embed_dim = teacher.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
            
            classifier = LinearClassifier(embed_dim, num_labels=num_labels)
            classifier = classifier.cuda()
            # classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[0])
            if args.dataset_name == 'ILSVRC':
                load_pretrained_linear_weights(classifier, args.arch, args.patch_size)
            elif args.dataset_name == 'Ericsson':
                state_dict_classifier = torch.load('linear_classifier_on_ericsson_dataset_patch_16/checkpoint.pth.tar')['state_dict']
                state_dict_classifier = {k.replace("module.", ""): v for k, v in state_dict_classifier.items()}
                classifier.load_state_dict(state_dict_classifier)
            elif args.dataset_name == 'CUB':
                torch_load_path_cub_cls_lyer = 'linear_classifier_on_CUB_dataset_patch_16-pretrained_dino/checkpoint.pth.tar' if args.dino_pretrained == '' else 'linear_classifier_on_CUB_dataset_patch_16/checkpoint.pth.tar'
                state_dict_classifier = torch.load(torch_load_path_cub_cls_lyer)['state_dict']
                state_dict_classifier = {k.replace("module.", ""): v for k, v in state_dict_classifier.items()}
                classifier.load_state_dict(state_dict_classifier)
        elif args.evaluate and args.classifier_for_top1_top5 == 'TransFg':
            from TransFG.train import setup as TransFG_setup
            _, classifier = TransFG_setup(utils.Dict2Obj({'model_type' : 'ViT-B_16', 'split':'overlap', 'slide_step':12, 'img_size':448, 
            'pretrained_dir':'/home/emursha/TransFG/pre_trained/ViT-B_16.npz', 
            'pretrained_model': '/home/emursha/TransFG/output/CUB_training_checkpoint.bin',
            'device':'cuda',
            'dataset': args.dataset_name,
            'smoothing_value':0.0}))
        else:
            classifier = None
            # raise ValueError("Invalid Classifier's Choice")
    else:
        classifier = None

    if classifier != None:
        classifier.train(False)
        classifier.eval()
        classifier = classifier.cuda()


        for p in classifier.parameters():
            p.requires_grad = False

    
    if args.evaluate or args.evaluate_only_on_attetion_maps:
        

        args.cam_curve_interval = 0.001
        if not args.evaluate_only_on_attetion_maps:
            fusioscore_net.load_state_dict(torch.load(os.path.join(args.evaluate_checkpoint, 'best_loc_acc_checkpoint_fusion.pth')))
        test_acc = evaluate_fusionmodule(-1, 'test', teacher, classifier, loaders, fusioscore_net, args, evaluate_on_MaxBoxV2 = False)
        return

    # ============ preparing optimizer ... ============
    params_groups = fusioscore_net.parameters()#utils.get_params_groups(fusioscore_net)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=args.lr, momentum=0.9, nesterov=True)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
   
    num_gpus = 1
    

    lr_schedule = lrs.StepLR(
        optimizer,
        step_size=args.lr_decay,
        gamma=args.gamma
    )

    print(f"Loss, optimizer and schedulers ready.")

    to_restore = {"epoch": 0, 'best_eval_acc': 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        fusioscore_net=fusioscore_net,
        optimizer=optimizer,
        lr_schedule=lr_schedule
    )
    start_epoch = to_restore["epoch"]
    best_eval_acc = to_restore["best_eval_acc"]

    start_time = time.time()
    # best_loc_acc = 0
    # with open(os.path.join(args.log_folder,'test.npy'), 'wb') as f:
    #     np.save(f, np.array(0))
    with (Path(args.output_dir) / "args.txt").open("a") as f:
        args_for_dump = copy.deepcopy(args)
        args_for_dump.api = None
        f.write(json.dumps(vars(args_for_dump)) + "\n")
        trial.log_parameters(args_for_dump)
        del args_for_dump

    weight_plot_dir = os.path.join(args.output_dir, 'weight_plots')
    if not os.path.exists(weight_plot_dir):
        os.makedirs(weight_plot_dir)

    loss_track_epochs = []
    loss_track_epochs_vline = [0]

    
    if args.seed_type == "MBProbSeederSLCAMS":
        seed_method = MBProbSeederSLCAMS()
    elif args.seed_type == "MBSeederSLCAMS":
        seed_method = MBSeederSLCAMS(min_p=args.seed_min_p)
    elif args.seed_type == "MBProbSeederSLCamsWithROI":
        seed_method = MBProbSeederSLCamsWithROI(min_ = args.seed_min, max_ = args.seed_max, min_p = args.seed_min_p, max_p = args.seed_max_p, seed_tech=args.seed_tech, bg_seed_tech=args.seed_tech_bg)
    with trial.train():
        for epoch in range(start_epoch, args.epochs):

            # ============ train for one epoch ... ============
            if args.is_certain_to_uncertain_train and epoch < len(args.certain_to_uncertain_th):
                if 'train_loader' in locals():
                    train_loader.dataset.uncertain_th = args.certain_to_uncertain_th[epoch]
                    del train_loader
                train_loader = get_data_loader_overlapped(
                    data_roots=args.data_paths,
                    metadata_root=args.metadata_root,
                    batch_size=args.batch_size_per_gpu,
                    workers=args.num_workers,
                    resize_size=args.resize_size,
                    crop_size=args.crop_size,
                    proxy_training_set=args.proxy_training_set,
                    num_val_sample_per_class=args.num_val_sample_per_class, 
                    args=args, 
                    overlapped_threshold=args.certain_to_uncertain_th[epoch])['train_fusion']
                print(f'Number of files @ epoch {epoch} @ th {args.certain_to_uncertain_th[epoch]} with the overlapped bboxs: {len(train_loader.dataset)}')
            else:
                train_loader = data_loader

            train_stats, loss_track = train_one_epoch(teacher, train_loader, optimizer, lr_schedule,
                                        epoch, fusioscore_net, classifier, class_loss, seed_loss,  seed_method, args, crf_loss)
            eval_acc = evaluate_fusionmodule(epoch, 'val', teacher, None, loaders, fusioscore_net, args)
            trial.log_metric("lr", lr_schedule.get_last_lr()[0], step=epoch)
            trial.log_metric("MaxBoxAcc", eval_acc, step=epoch)

            trial.log_metric("total_loss", train_stats['loss'], step=epoch)
            # trial.log_metric("cls_loss", train_stats['cls_loss'], step=epoch)
            # trial.log_metric("seed_loss", train_stats['seed_loss'], step=epoch)
            # ============ writing logs ... ============
            best_save_acc = copy.deepcopy(best_eval_acc)
            if eval_acc > best_save_acc:
                best_save_acc = eval_acc
            save_dict = {
                'fusioscore_net': fusioscore_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': lr_schedule.state_dict(),
                'epoch': epoch + 1,
                'best_eval_acc': best_save_acc,
                'args': args,
            }

           
            torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            if args.save_checkpoint_each_epoch:
                torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pth'))
            if eval_acc > best_eval_acc:
                torch.save(fusioscore_net.state_dict(), os.path.join(args.output_dir, f'best_loc_acc_checkpoint_fusion.pth'))
                best_eval_acc = eval_acc

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

           
            with open(os.path.join(args.output_dir, f'plot_data.pickle'), 'wb') as handle:
                plot_data = {'loss_track_epochs': loss_track_epochs, 'loss_track_epochs_vline': loss_track_epochs_vline}
                pickle.dump(plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    with trial.test():
        fusioscore_net.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_loc_acc_checkpoint_fusion.pth')))
        test_acc = evaluate_fusionmodule(-1, 'test', teacher, None, loaders, fusioscore_net, args)
        trial.log_metric("MaxBoxAcc", test_acc, step=0)
        trial.log_other("status", "done")
        trial.end()
    # wandb.log({"test_MaxBoxAcc": test_acc})

    print('Training time {}'.format(total_time_str))

    return test_acc

def train_one_epoch(teacher, data_loader, optimizer, lr_schedule,
                    epoch, fusioscore_net, classifier, class_loss, seed_loss, seed_method, args, crf_loss):
    
    fusioscore_net.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    loss_track = []
    for it, (images, raw_images, target, _, roi, head_indx) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        # for i, param_group in enumerate(optimizer.param_groups):
        #     param_group["lr"] = lr_schedule[it]

        images = images.cuda()
        target = target.cuda()

        image_size = images.shape[2:]
        w_featmap = images.shape[-2] // teacher.patch_embed.patch_size
        h_featmap = images.shape[-1] // teacher.patch_embed.patch_size
        # img_size = images.shape[-2]
        attentions = teacher.get_last_selfattention(images)
        nh = attentions.shape[1]  # number of head
        attentions = attentions[:,:, 0, 1:].reshape(-1, nh, w_featmap, h_featmap)#attentions.reshape(-1, nh, w_featmap, h_featmap)
        attentions = torch.nan_to_num(attentions, nan=0.0, posinf=1., neginf=0.0)
        attentions = nn.functional.interpolate(attentions, size=(args.crop_size,args.crop_size), mode="bilinear", align_corners=False)
        # attentions = normalize_tensor(attentions)
        # attentions = attentions
        if args.seed_type != 'MBProbSeederSLCamsWithROI':
            # attentions = normalize_tensor(attentions)
            mean_attention = torch.mean(attentions, dim=1).unsqueeze(1)
            mean_attention = normalize_tensor(mean_attention)
            final_attention = mean_attention
            seed_attentions = seed_method(mean_attention.detach())
        else:
            score_maps = utils.get_maps_for_binary_regions(attentions, args.evaluation_type)
            final_attention = score_maps[torch.arange(score_maps.size(0)), head_indx, :, :].unsqueeze(1).detach()
            if args.normalize_with_softmax:
                img_shape = final_attention.shape
                new_normalization = rearrange(F.softmax(rearrange(final_attention*args.normlization_temprature, 'b c h w -> b (c h w)')), 'b (c h w) -> b c h w', c=img_shape[1], h=img_shape[2], w=img_shape[3])
            else:
                new_normalization = normalize_tensor(final_attention)

            seed_attentions = seed_method(new_normalization.detach(), roi.unsqueeze(1).cuda())
            del new_normalization
            
            if args.dataset_name == 'Ericsson' and args.seed_type == 'MBProbSeederSLCamsWithROI' and args.use_full_bg_with_no_fg_regions:
                for tmp_i, tmp_target in enumerate(target):
                    if tmp_target == 0:
                        seed_attentions[tmp_i] *= 0
        # attentions = Variable(attentions.data, requires_grad = True)
        attentions = attentions[:, :args.number_of_heads_for_fusion, :, :]
        if args.use_image_as_input_for_fusion:
            attentions = torch.cat([attentions, images], axis=1)
        # del images

        if args.test_with_synth_inp:
            new_attn = (attentions*0)
            new_attn[:, :, 0:20, 0:20] = 0.5
        else:
            new_attn = attentions

        cam_fused = fusioscore_net(new_attn.detach())

        torch.cuda.empty_cache()

        # logits = resnet_classifier(torch.mul(final_attention, images))#resnet_classifier(attentions)
        # loss_cls = class_loss(logits, target)
        if args.seperate_map_for_each_class:
            for ind in range(len(target)):
                seed_attentions[ind] = torch.where(seed_attentions[ind] == 1, target[ind]+1, seed_attentions[ind])
            # seed_attentions_tmp = (torch.ones([seed_attentions.shape[0], args.num_labels, seed_attentions.shape[1], seed_attentions.shape[2]], dtype=torch.int64)*args.seed_seg_ignore_idx).cuda()
            # seed_attentions_tmp[torch.arange(score_maps.size(0)), target, :, :] = seed_attentions
        loss_seed = seed_loss(cam_fused, seed_attentions.detach())

        if args.use_classifier or args.crf_fc:
            loss_seed = args.lmda_seed_loss * loss_seed
            loss = loss_seed #args.lmda_seed_loss * loss_seed
            if args.use_classifier:
                if not args.use_dino_classifier:
                    logits = classifier(torch.mul(final_attention, images))
                    if args.dataset_name == 'OpenImages':
                        logits = logits['logits']
                else:
                    intermediate_output = teacher.get_intermediate_layers(torch.mul(final_attention, images), args.n_last_blocks)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if args.avgpool_patchtokens:
                        output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        output = output.reshape(output.shape[0], -1)
                    logits = classifier(output)
                loss_cls = class_loss(logits, target)
                loss_cls = args.lmda_clsloss * loss_cls
                loss += loss_cls

            if args.crf_fc:
                loss_crf = crf_loss(fcams=cam_fused, raw_img=raw_images.type(torch.uint8))
                loss += loss_crf

        else:
            loss = loss_seed

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss_track.append([loss_cls.item(), loss_seed.item(), loss.item()])

        metric_logger.update(loss=loss.item())
        if args.use_classifier or args.crf_fc:
            metric_logger.update(loss_seed=loss_seed.item())
            if args.use_classifier:
                metric_logger.update(loss_cls=loss_cls.item())
            if args.crf_fc:
                metric_logger.update(loss_crf=loss_crf.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    print("Averaged stats:", metric_logger)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if lr_schedule.get_last_lr()[0] >= 1e-8:
        lr_schedule.step()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, loss_track

def get_params(exp, param_name):
  best_test_Params = exp.get_parameters_summary(param_name)
  assert best_test_Params['valueMax'] == best_test_Params['valueMin'] == best_test_Params['valueCurrent'], 'Invalid params value'
  return best_test_Params['valueMax']

def run_main(trial, args):
    tags = [args.dataset_name, args.seed_type, args.arch, args.patch_size, args.iou_regions_path.split('/')[-2].rsplit('_',1)[0], 
    f"FusionModule_on_{args.dataset_name}_for_{args.seed_type}_with_{args.arch}_{args.patch_size}_{args.iou_regions_path.split('/')[-2].rsplit('_',1)[0]}"]
    
    if args.search_hparameters:
        args.lr = trial.params["lr"]
        if args.use_classifier:
            args.lmda_clsloss = trial.params['lmda_clsloss']
            args.lmda_seed_loss = trial.params['lmda_seed_loss']
        if args.top_n_head_for_sampling > 0:
            args.top_n_head_for_sampling = trial.params["top_n_head_for_sampling"]

        if args.seed_type == "MBSeederSLCAMS" or args.seed_type == "MBProbSeederSLCamsWithROI":
            args.seed_min_p= trial.params["seed_min_p"]
            args.seed_max_p= trial.params["seed_max_p"]
        if args.seed_type == "MBProbSeederSLCamsWithROI":
            args.seed_min = trial.params['seed_min']
            args.seed_max = trial.params['seed_max']
        if args.normalize_with_softmax and args.seed_type == "MBProbSeederSLCamsWithROI":
            args.normlization_temprature = trial.params["normlization_temprature"]
        print(f'Searching Hyper Paramter on {args.dataset_name} with study name {args.study_name}')
        args.experiment_name = f'trial_{str(trial.id)}'

    elif args.run_for_best_setting:
        raise NotImplemented('Not Implemented fro best setting')

    else:
        raise ValueError("Invalid option specified")


    args.log_folder = configure_log_folder(args)
    args.data_paths = configure_data_paths(args)
    args.output_dir = args.log_folder
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("config:", args)

    trial.add_tags(tags)
    trial.log_other("status", "running")

    return train_fusionmodule(args, trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FusionModule', parents=[get_args_parser()])
    args = parser.parse_args()

    comet_workspace = '' #replace with your comet workspace
    
    api_key = '' #replace with your comet api key
    api = comet_ml.api.API(api_key=api_key)
    args.api = api

    if args.evaluate or args.run_for_given_setting or args.evaluate_only_on_attetion_maps:
        if os.path.exists(args.evaluate_checkpoint) or args.evaluate_only_on_attetion_maps:
            if os.path.exists(args.evaluate_checkpoint):
                keys_to_ignore = ['iou_regions_path', 'num_samples_to_save', 'mask_root', 'metadata_root', 'evaluate_only_on_attetion_maps', 'exp_pre_name', 'classifier_for_top1_top5', 'run_accumulate_and_save_fun', 'proxy_training_set', 'evaluate', 'run_for_given_setting', 'evaluate_checkpoint', 'data_root']
                with open(os.path.join(args.evaluate_checkpoint, 'args.txt')) as json_file:
                    args_tmp = json.load(json_file)
                    for key in args_tmp.keys():
                        if 'dino_pretrained' == key and args_tmp[key] == '':
                            setattr(args, key, '')
                        elif 'dino_pretrained' == key and args_tmp[key] != '':
                            if os.path.exists(args_tmp[key]):
                                setattr(args, key, args_tmp[key])
                            elif os.path.exists(getattr(args, 'dino_pretrained')):
                                pass
                            else:
                                raise ValueError('Invalid DINO Pre-trained Path')
                        elif key not in keys_to_ignore:
                            setattr(args, key, args_tmp[key])
                args.old_experiment_name = os.path.join(os.path.split(args.experiment_category)[-1], args.experiment_name)

            if args.evaluate:
                args.experiment_category = f'evaluate_on_best_trials_{args.exp_pre_name}' 
            elif args.run_for_given_setting:
                 args.experiment_category = f'rerun_exp_with_same_setting_{args.exp_pre_name}'
            else:
                args.experiment_category = f'evaluate_on_six_channels_{args.exp_pre_name}'
            trial = None if (args.evaluate or args.evaluate_only_on_attetion_maps) else Experiment(api_key=api_key, project_name=args.experiment_category[:95], workspace=comet_workspace, log_code = False)
            
            if not hasattr(args, 'old_experiment_name'):
                args.old_experiment_name = f'{args.dataset_name}_{str(uuid.uuid4())}'
            args.experiment_name = args.old_experiment_name if trial == None else f'best_trial_{str(trial.id)}'

            args.data_paths = configure_data_paths(args)
            args.log_folder = configure_log_folder(args)
            args.output_dir = args.log_folder

            train_fusionmodule(args, trial)
        else:
            raise ValueError('Invalid Checkpoint path to evaluate')

    else:
        if args.search_hparameters == args.run_for_best_setting and (args.search_hparameters or args.run_for_best_setting):
            raise ValueError("Hyperparamter search and Run for best setting can't be selected at the same time")

        if args.search_hparameters or args.run_for_best_setting:
            pretrained_singal = '' if args.dino_pretrained == '' else 'p'
            study_name=f"Fsion_cmt_{args.dataset_name}_{pretrained_singal}_{args.exp_pre_name}_on_{args.iou_regions_path.split('/')[-2].rsplit('_',1)[0]}"
            if args.test_with_synth_inp:
                study_name = 'syninp_' + study_name

            study_name = study_name[:95]

            args.experiment_category = os.path.join(args.output_dir, study_name)
            os.makedirs(args.experiment_category, exist_ok=True)
            os.makedirs(study_name, exist_ok=True)

            comet_projectname = study_name#args.experiment_category

            if args.search_hparameters:   
                comet_config_file_path = os.path.join(study_name, 'comet_config.txt')
                if os.path.exists(comet_config_file_path):
                    with open(comet_config_file_path) as f:
                        comet_config_id = f.readline()
                    opt = Optimizer(comet_config_id)
                else:
                    lr_search_space = {"type": "discrete", "values":[0.1, 0.01, 0.001, 0.0001]} if args.discrete_search_space else {"type": "float", "min": 0.000001, "max": 0.1}
                    paramters = {#gamma 0.1, epochs=3 #STEPLR
                    "lr": lr_search_space,}

                    if args.use_classifier:
                        paramters["lmda_clsloss"] = {"type": "discrete", "values": [0, 0.001, 0.1, 0.3, 0.6, 1]} if args.discrete_search_space else {"type": "float", "min": 0.0, "max": 1.0}
                        paramters["lmda_seed_loss"] = {"type": "discrete", "values": [0, 0.1, 0.3, 0.6, 1,]} if args.discrete_search_space else {"type": "float", "min": 0.0, "max": 1.0}

                    # if args.crf_fc:
                    #     paramters["lmda_crf_fc"] = {"type": "discrete", "values": [1, 0.6, 0.3, 0.1, 0.001, 0.0001, 0.000001, 0.00000001, 0.0000000001]}
                    
                    if args.seed_type == "MBSeederSLCAMS" or args.seed_type == "MBProbSeederSLCamsWithROI":
                        seed_min_max_p_search_space = {"type": "discrete", "values": [.005, .05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60]} if args.discrete_search_space else {"type": "float", "min": 0.0, "max": 1.0}#[.005, .01, .03, .05] if (args.dataset_name == "Ericsson" or args.dataset_name == "CUB") else [.01, .03]
                        paramters["seed_min_p"] = seed_min_max_p_search_space
                        paramters["seed_max_p"] = seed_min_max_p_search_space
                    if args.seed_type == "MBProbSeederSLCamsWithROI": #and (args.dataset_name == "Ericsson" or args.dataset_name == "CUB"):
                        seed_min_max_search_space =  {"type": "discrete", "values":[1, 10, 50, 100, 1000, 10000]} if args.discrete_search_space else {"type": "integer", "min": 1, "max": 10000}
                        paramters["seed_min"] =  seed_min_max_search_space
                        paramters["seed_max"] = seed_min_max_search_space

                    if args.normalize_with_softmax and args.seed_type == "MBProbSeederSLCamsWithROI":
                        paramters["normlization_temprature"] = {"type": "discrete", "values": [5, 10, 30]}

                    if args.top_n_head_for_sampling > 0:
                        paramters["top_n_head_for_sampling"] = {"type": "discrete", "values": [3, 5, 7]} if args.discrete_search_space else {"type": "integer", "min": 1, "max": 10}

                    config = {
                        "algorithm": args.hp_search_type,
                        "spec": {
                        "objective": "maximize",
                        # "retryAssignLimit": 5,
                        "metric": "test_MaxBoxAcc", },
                        "parameters": paramters,
                    }
                    opt = Optimizer(config)
                    with open(comet_config_file_path, "w") as f:
                        f.write(opt.id)
                    with open(os.path.join(study_name, 'comet_config_params.txt'), "w") as f:
                        f.write(json.dumps(paramters) + "\n")
                args.opt_id = opt.id

            if args.only_create_study:
                print('study created succesfully. Now exiting...')
                quit()

            args.study_name = study_name

            if args.search_hparameters:
                # optimize_fusion_module_lmbda = lambda trial: run_main(trial, args)
                # study.optimize(optimize_fusion_module_lmbda, n_trials=1)
                
                # if not args.import_from_optuna:
                trial = opt.next(api_key=api_key, project_name=comet_projectname, workspace=comet_workspace,
                log_code = False)
                run_main(trial, args)
                # else:
                #     trial_to_test = 0
                #     for trial in opt.get_experiments(api_key=api_key, project_name=comet_projectname, workspace=comet_workspace):
                #         print('(N) tested trials', trial_to_test)
                #         trial_to_test += 1
                #         # run_main(trial, args)
            elif args.run_for_best_setting:
                run_main(trial = None, args = args)