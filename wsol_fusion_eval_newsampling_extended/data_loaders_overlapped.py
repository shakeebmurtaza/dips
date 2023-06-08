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

import munch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
from PIL import ImageFilter, ImageOps
import pickle
import utils
import utils_tranforms_with_two_target as tranforms_with_target
import gc
from glob import glob
from matplotlib import pyplot as plt
_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = ['train_fusion']#, 'train', 'val', 'test')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = os.path.join(metadata_root,
                                            'image_ids_proxy.txt')
    metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    metadata.localization = os.path.join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
            x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes


def get_mask_paths(metadata):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes

class WSOLImageLabelDataset(Dataset):
    def __init__(self, data_root, metadata_root, transform, proxy,
                 num_sample_per_class=0, args=None, resize_size=None, crop_size=None, 
                 overlapped_threshold=None):
        self.list_imgs_with_best_head = []
        self.args =args
        if 'train_fusion' in metadata_root:
            if args.dataset_name == 'ILSVRC' or args.dataset_name == 'Ericsson' or args.dataset_name == 'CUB':
                meta_data_path = glob(os.path.join(args.is_certain_to_uncertain_train_iou_regions_root, f'*{str(overlapped_threshold).replace(".", "-")}_*'))
                assert len(meta_data_path) == 1, 'Invalid number of meta data files'
                gc.disable()
                file = open(os.path.join(meta_data_path[0], '1_4_heads_and_mean_on_train_list_imgs_with_best_head_only_headwise_bbox.pkl'), 'rb')
                self.meta_data = pickle.load(file)
                file.close()
                gc.enable()

                meta_data_keys = list(self.meta_data.keys())
                # meta_data_keys.remove('area_sucedded_for_n_imgs')
                # meta_data_keys.remove('area_sucedded_for_n_imgs_percentage')
                for indx, key in enumerate(meta_data_keys):
                    if self.meta_data[key]['lst_head_wise_newbbox'] != None:
                        self.list_imgs_with_best_head.append(key)

                transform = tranforms_with_target.Compose([
                    tranforms_with_target.Resize((resize_size, resize_size)),
                    tranforms_with_target.RandomCrop(crop_size),
                    tranforms_with_target.RandomHorizontalFlip(),
                    tranforms_with_target.ToTensor(),
                    tranforms_with_target.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
                ])
        if 'train_fusion' in metadata_root:
            metadata_root = metadata_root.replace('train_fusion','train')
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = self.list_imgs_with_best_head#get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class

        self._adjust_samples_per_class()

        if args.classifier_for_top1_top5 != None:
            self.extra_test_transform=transforms.Compose([transforms.Resize((448, 448), Image.BILINEAR),
                                    # transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)])


    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')

        if self.list_imgs_with_best_head != None:
            list_map = []
            for bbox_for_fill in self.meta_data[image_id]['lst_head_wise_newbbox']:
                map = np.zeros([np.asarray(image).shape[0], np.asarray(image).shape[1]])
                # bbox_for_fill = np.asarray(self.list_imgs_with_best_head[image_id]['newbbox']).astype(np.int32)
                box_fused_input = utils.draw_bbox(map.astype('uint8'), np.asarray(bbox_for_fill).reshape(1, 4).astype(np.int32))
                box_fused_input = utils.fill_bounding_box(box_fused_input)
                box_fused_input = np.asarray(box_fused_input)
                box_fused_input = np.where(box_fused_input == 255, 1, 0).astype('uint8')
                list_map.append(box_fused_input)
            overlap_map = np.asarray(list_map)
            overlap_map = np.prod(overlap_map, axis=0)
            
            # for ind, map in enumerate(list_map):
            #     plt.subplot(1, 6, ind+1)
            #     plt.imshow(map)
            #     plt.axis('off')
            # plt.subplot(1, 6, 6)
            # plt.imshow(overlap_map)
            # plt.axis('off')
            # plt.savefig('test.png')
            overlap_map = Image.fromarray(overlap_map.astype('uint8'))

            image, box_fused_input, raw_image = self.transform(image, overlap_map, image)
            return image, raw_image, image_label, image_id, box_fused_input, np.asarray(self.meta_data[image_id]['head_index'])

        image_new_test_transformed = self.extra_test_transform(image) if self.args.classifier_for_top1_top5 != None else torch.zeros(0)
        image = self.transform(image)
        return image, image_new_test_transformed, image_label, image_id

    def __len__(self):
        return len(self.image_ids)

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

def get_data_loader(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set,
                    num_val_sample_per_class=0, args=None):
    dataset_transforms = dict(
        train= DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    ),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        train_fusion=transforms.Compose([
                    transforms.Resize((resize_size, resize_size)),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
                ]),
    )
    # dataset_transforms = dict(
    #     train=transforms.Compose([
    #         transforms.Resize((resize_size, resize_size)),
    #         transforms.RandomCrop(crop_size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]),
    #     val=transforms.Compose([
    #         transforms.Resize((crop_size, crop_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]),
    #     test=transforms.Compose([
    #         transforms.Resize((crop_size, crop_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]))

    loaders = {
        split: DataLoader(
            WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0)
            ),
            sampler=torch.utils.data.DistributedSampler(WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0)
            ), shuffle=True),
            batch_size=batch_size,
            # shuffle=split == 'train',
            num_workers=workers)
        for split in _SPLITS
    }
    return loaders

def get_data_loader_Fusion(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set,
                    num_val_sample_per_class=0, args=None):
    dataset_transforms = dict(
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        train=transforms.Compose([
                    transforms.Resize((resize_size, resize_size)),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
                ]),
    )
    # dataset_transforms = dict(
    #     train=transforms.Compose([
    #         transforms.Resize((resize_size, resize_size)),
    #         transforms.RandomCrop(crop_size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]),
    #     val=transforms.Compose([
    #         transforms.Resize((crop_size, crop_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]),
    #     test=transforms.Compose([
    #         transforms.Resize((crop_size, crop_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]))

    loaders = {
        split: DataLoader(
            WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0)
            ),
            sampler=torch.utils.data.DistributedSampler(WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0)
            ), shuffle=True),
            batch_size=batch_size,
            # shuffle=split == 'train',
            num_workers=workers)
        for split in ('train', 'val', 'test')
    }
    return loaders


def get_data_loader_non_dist(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set,
                    num_val_sample_per_class=0, args=None, overlapped_threshold=None):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        train_fusion=transforms.Compose([
                    transforms.Resize((resize_size, resize_size)),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
                ]),
    )
    # dataset_transforms = dict(
    #     train=transforms.Compose([
    #         transforms.Resize((resize_size, resize_size)),
    #         transforms.RandomCrop(crop_size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]),
    #     val=transforms.Compose([
    #         transforms.Resize((crop_size, crop_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]),
    #     test=transforms.Compose([
    #         transforms.Resize((crop_size, crop_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    #     ]))

    loaders = {
        split: DataLoader(
            WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and (split == 'train' or split == 'train_fusion'),
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0),
                args = args,
                resize_size=resize_size, 
                crop_size=crop_size,
                overlapped_threshold=overlapped_threshold
            ),
            # shuffle=True,
            batch_size=batch_size//2 if split == 'test' else batch_size,
            shuffle=(split == 'train' or split == 'train_fusion'),
            num_workers=workers)
        for split in _SPLITS
    }
    return loaders

if __name__ == "__main__":
    from os.path import join as ospj
    from utils import Dict2Obj

    train = val = test = ospj('/home/emursha/datasets', 'ILSVRC')
    data_paths = mch(train=train, val=val, test=test, train_fusion=train)
    args = Dict2Obj({"data_paths":data_paths,
                     "batch_size_per_gpu":32,
                     "num_workers":4,
                     "resize_size":256,
                     "crop_size":224,
                     "proxy_training_set":False,
                     "num_val_sample_per_class":0,
                     "overlapped_threshold":0.85,
                     "num_classes":1000,
                     "metadata_root":'/home/emursha/DGA1032_code/dino_orginal/metadata/ILSVRC',
                     "classifier_for_top1_top5": None,
                     "dataset_name": "ILSVRC",
                     "is_certain_to_uncertain_train": True,
                     "is_certain_to_uncertain_train_iou_regions_root": "Extract_bboxs_overlapped__ILSVRC_with_vit_small_16_for_class_probability_with_dino_clfr_False",
                     "certain_to_uncertain_th": [.8, .75, .7, .65, .6, .55, .5, .45, .4, .35]
                     })

    for epoch in range(15):
        if args.is_certain_to_uncertain_train and epoch < len(args.certain_to_uncertain_th):
            train_loader = get_data_loader_non_dist(
                            data_roots=args.data_paths,
                            metadata_root=args.metadata_root,
                            batch_size=1,
                            workers=args.num_workers,
                            resize_size=args.resize_size,
                            crop_size=args.crop_size,
                            proxy_training_set=args.proxy_training_set,
                            num_val_sample_per_class=args.num_val_sample_per_class, 
                            args=args, 
                            overlapped_threshold=args.certain_to_uncertain_th[epoch])['train_fusion']
            print(f'Number of files @ epoch {epoch} @ th {args.certain_to_uncertain_th[epoch]} with the overlapped bboxs: {len(train_loader.dataset)}')
            if epoch == 0:
                for instance in train_loader:
                    pass
        else:
            print(f'training with full dataset @ epoch {epoch}')
    
    print('')