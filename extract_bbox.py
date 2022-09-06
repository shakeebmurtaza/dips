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
import argparse
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import pickle as pkl
import optuna
import gc

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from wsol_eval_extract_bbox.data_loaders import get_data_loader
from wsol_eval_extract_bbox.wsol_utils import str2bool, configure_data_paths, configure_log_folder
from torchvision import models
import uuid

from wsol_eval_extract_bbox.inference import CAMComputer
from wsol_eval_extract_bbox.wsol.resnet import resnet50 as resnet50_wsol_eval

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

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
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
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
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
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
    # parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
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
    _DATASET_NAMES = ('CUB', 'ILSVRC', 'Ericsson', 'OpenImages')
    parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='dataset/',
                        help='path to dataset images')
    #Eval WSOL Paramters
    parser.add_argument('--cam_curve_interval', type=float, default=0.01,
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
                        help="Root folder of masks (OpenImages).")
    parser.add_argument('--patch_indx', type=int, default='0')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--experiment_category', type=str, default='train_log/',
                        help="Root folder for saving experiemnt detials.")

    parser.add_argument('--evaluation_type', default='all_heads', type=str,
        choices=['all_heads', 'all_heads_and_mean', '1_5_heads_and_mean', '1_4_heads_and_mean', 
        '1_5_heads_and_mean_1_5', '1_4_heads_and_mean_1_4', 'all_heads_and_mean_1_4',
        'mean_1_2_mean_1_3_mean_1_4_mean_1_5'])
    parser.add_argument('--score_calulcation_method', default='class_probability', type=str,
        choices=['class_probability', 'size_probability', 'size_and_class_probability'])

    parser.add_argument('--num_samples_to_save',
                        type=int, default=0)

    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")

    parser.add_argument('--use_dino_classifier', default=False, type=utils.bool_flag)

    parser.add_argument('--bbox_min_size', type=float, default=0.0,
                        help="At which threshold intervals will the score maps "
                             "be evaluated?.")

    parser.add_argument('--only_create_study', type=str2bool, nargs='?',
        const=True, default=False)

    parser.add_argument('--search_hparameters', type=str2bool, nargs='?',
                        const=True, default=True)

    parser.add_argument('--nested_contours', type=str2bool, nargs='?',
        const=True, default=False)

    parser.add_argument('--save_final_cams', type=str2bool, nargs='?',
                        const=True, default=False)

    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')

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

def test_dino(trial, args):
    if trial != None:
        args.bbox_min_size = trial.suggest_categorical('bbox_min_size', args.search_space['bbox_min_size'])
        args.experiment_name = f'trial_{str(trial._trial_id)}'
    else:
        args.experiment_name = f'{args.score_calulcation_method}_dino_{str(args.use_dino_classifier)}_th{str(args.bbox_min_size).replace(".", "-")}_nestedcontor_{str(args.nested_contours)}_split_{args.split}_proxy{int(args.proxy_training_set)}'
        if args.save_final_cams:
            args.experiment_name += f'_f_c{str(int(args.save_final_cams))}'
        args.experiment_name += 'with_bg_rois'
        args.experiment_name += f'_{str(uuid.uuid4())}'
    args.log_folder = configure_log_folder(args)
    args.data_paths = configure_data_paths(args)
   
    loaders = get_data_loader(
        data_roots=args.data_paths,
        metadata_root=args.metadata_root,
        batch_size=args.batch_size_per_gpu,
        workers=args.num_workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        proxy_training_set=args.proxy_training_set,
        num_val_sample_per_class=args.num_val_sample_per_class, args=args)

    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit"):
        student = torch.hub.load('facebookresearch/xcit', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    utils.load_pretrained_weights(teacher, args.dino_pretrained, 'teacher', args.arch, args.patch_size)
    teacher = teacher.cuda()

    if not args.use_dino_classifier:
        if args.dataset_name == 'ILSVRC':
            classifier = models.resnet50(pretrained=True).cuda()
        elif args.dataset_name == 'OpenImages':
            classifier = resnet50_wsol_eval('cam', num_classes=args.num_labels, large_feature_map=False, pretrained=False)#models.resnet50(pretrained=True).cuda()
            classifier_weight_path = 'wsol_eval_extract_bbox/wsol/last_checkpoint_resnet_openimages.pth.tar'
            state_dict = torch.load(classifier_weight_path)
            classifier.load_state_dict(state_dict['state_dict'])
            print("classifier's weights loaded from {}".format(classifier_weight_path))
            classifier = classifier.cuda()
    else:
        embed_dim = teacher.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
        classifier = classifier.cuda()
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

    evaluate(-1, 'test', teacher, loaders, args.dino_pretrained, classifier, args)
    evaluate(-1, 'train', teacher, loaders, args.dino_pretrained, classifier, args)
    return evaluate(-1, 'val', teacher, loaders, args.dino_pretrained, classifier, args)

def evaluate(epoch, split, model, loaders, experiment_name, classifier, args):
    print("Evaluate epoch {}, split {}".format(epoch, split))
    model.eval()

    with (Path(args.log_folder) / f"args_on_{split}.txt").open("a") as f:
        f.write(json.dumps(vars(args)) + "\n")

    cam_computer = CAMComputer(
        model=model,
        loader=loaders[split],
        metadata_root=os.path.join(args.metadata_root, split),
        mask_root=args.mask_root,
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=args.dataset_name,
        split=split,
        cam_curve_interval=args.cam_curve_interval,
        multi_contour_eval=False,#args.multi_contour_eval,
        log_folder=args.log_folder,
        patch_indx = args.patch_indx,
        evaluation_type = args.evaluation_type,
        classifier = classifier,
        num_samples_to_save = args.num_samples_to_save,
        intermidiate_pkl_file = os.path.join(args.log_folder, f"{args.evaluation_type}_on_{split}_list_imgs_with_best_head_intermediate.pkl"),
        args = args
    )
    cam_performance, list_imgs_with_best_head = cam_computer.compute_and_evaluate_cams()
    if cam_performance != None:
        cam_performance["patch_id"] = args.patch_indx
        cam_performance["split"] = split
        cam_performance["exp_name"] = experiment_name
        with (Path(args.log_folder) / f"{args.evaluation_type}_on_{split}_log.txt").open("a") as f:
            f.write(json.dumps(cam_performance) + "\n")
        print(cam_performance)

    

    file = open(os.path.join(args.log_folder, f"{args.evaluation_type}_on_{split}_list_imgs_with_best_head.pkl"),'wb+')
    pkl.dump(list_imgs_with_best_head, file)
    file.close()

    if cam_performance != None:
        if args.dataset_name == 'OpenImages':
            return cam_performance[0]
        return cam_performance[0][1]
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()

    pretrained_singal = '' if args.dino_pretrained == '' else 'p'

    study_name=f"Extract_bboxs_overlapped_topn_{pretrained_singal}_{os.path.split(args.metadata_root)[-1]}_with_{args.arch}_{args.patch_size}_for_{args.score_calulcation_method}_with_dino_clfr_{str(args.use_dino_classifier)}"
    args.experiment_category = study_name
    if args.search_hparameters:
        args.search_space = {"bbox_min_size": [.5, .55, .6, .65, .7, .75]}#{"bbox_min_size": [.25, .3, .35, .4, .45]}#{"bbox_min_size": [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]}
        
        storage_path = f"optuna_studies/{study_name}.db"
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{storage_path}', engine_kwargs={"connect_args": {"timeout": 30000}})
        try:
            study = optuna.load_study(study_name=study_name, storage = storage, sampler=optuna.samplers.GridSampler(args.search_space))#="sqlite:///optuna_studies/ProbabilisticSampling.db")
        except KeyError:
            study = optuna.create_study(direction='maximize', study_name = study_name, storage=storage, sampler=optuna.samplers.GridSampler(args.search_space))

        if args.only_create_study:
            print('study created succesfully. Now exiting...')
            quit()

        args.study_name = study_name

        optimize_fusion_module_lmbda = lambda trial: test_dino(trial, args)
        study.optimize(optimize_fusion_module_lmbda, n_trials=1)
    else:
        test_dino(None, args)