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

import argparse
import cv2
import numpy as np
import os
from os.path import join as ospj
import torch.utils.data as torchdata
from copy import deepcopy
from .wsol_utils import str2bool
from .data_loaders import configure_metadata
from .data_loaders import get_image_ids
from .data_loaders import get_bounding_boxes
from .data_loaders import get_image_sizes
from .data_loaders import get_mask_paths
from .wsol_utils import check_scoremap_validity
from .wsol_utils import check_box_convention
from .wsol_utils import t2n
import utils
from matplotlib import pyplot as plt
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

# _IMAGENET_MEAN = [0.485, .456, .406]
# _IMAGENET_STDDEV = [.229, .224, .225]
# _RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, image_ids):
        self.scoremap_path = scoremap_path
        self.image_ids = image_ids

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, image_id + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        return cam, image_id

    def __len__(self):
        return len(self.image_ids)

# def cls_loc_err(topk_boxes, gt_label, gt_boxes, topk=(1,), iou_th=0.5):
#     assert len(topk_boxes) == len(topk)
#     gt_boxes = gt_boxes
#     gt_box_cnt = len(gt_boxes) // 4
#     topk_loc = []
#     topk_cls = []
#     for topk_box in topk_boxes:
#         loc_acc = 0
#         cls_acc = 0
#         for cls_box in topk_box:
#             max_iou = 0
#             max_gt_id = 0
#             for i in range(gt_box_cnt):
#                 gt_box = gt_boxes[i*4:(i+1)*4]
#                 iou_i = cal_iou(cls_box[1:], gt_box)
#                 if  iou_i> max_iou:
#                     max_iou = iou_i
#                     max_gt_id = i
#             if len(topk_box)  == 1:
#                 wrong_details = get_badcase_detail(cls_box, gt_boxes, gt_label, max_iou, max_gt_id)
#             if cls_box[0] == gt_label:
#                 cls_acc = 1
#             if cls_box[0] == gt_label and max_iou > iou_th:
#                 loc_acc = 1
#                 break
#         topk_loc.append(float(loc_acc))
#         topk_cls.append(float(cls_acc))
#     return topk_cls, topk_loc, wrong_details

def cal_iou(box1, box2, method='iou'):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    if method == 'iog':
        iou_val = i_area / (box2_area)
    elif method == 'iob':
        iou_val = i_area / (box1_area)
    else:
        iou_val = i_area / (box1_area + box2_area - i_area)
    return iou_val

# def get_badcase_detail(top1_bbox, gt_bboxes, gt_label, max_iou, max_gt_id):
def get_badcase_detail(pred_cls, pred_bbox, gt_bboxes, gt_label, max_iou, max_gt_id):
    cls_wrong = 0
    multi_instances = 0
    region_part = 0
    region_more = 0
    region_wrong = 0

    # pred_cls = top1_bbox[0]
    # pred_bbox = top1_bbox[1:]

    if not int(pred_cls) == gt_label:
        cls_wrong = 1
        return cls_wrong, multi_instances, region_part, region_more, region_wrong

    if max_iou > 0.5:
        return 0, 0, 0, 0, 0

    # multi_instances error
    # gt_box_cnt = len(gt_bboxes) // 4
    gt_box_cnt = len(gt_bboxes)
    if gt_box_cnt > 1:
        iogs = []
        for i in range(gt_box_cnt):
            # gt_box = gt_bboxes[i * 4:(i + 1) * 4]
            gt_box = gt_bboxes[i:i+1]
            iog = cal_iou(pred_bbox, gt_box, method='iog')
            iogs.append(iog)
        if sum(np.array(iogs) > 0.3)> 1:
            multi_instances = 1
            return cls_wrong, multi_instances, region_part, region_more, region_wrong
    # region part error
    # iog = cal_iou(pred_bbox, gt_bboxes[max_gt_id*4:(max_gt_id+1)*4], method='iog')
    iog = cal_iou(pred_bbox, gt_bboxes[max_gt_id:max_gt_id+1], method='iog')
    # iob = cal_iou(pred_bbox, gt_bboxes[max_gt_id*4:(max_gt_id+1)*4], method='iob')
    iob = cal_iou(pred_bbox, gt_bboxes[max_gt_id:max_gt_id+1], method='iob')
    if iob >0.5:
        region_part = 1
        return cls_wrong, multi_instances, region_part, region_more, region_wrong
    if iog >= 0.7:
        region_more = 1
        return cls_wrong, multi_instances, region_part, region_more, region_wrong
    region_wrong = 1
    return cls_wrong, multi_instances, region_part, region_more, region_wrong


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, dataset_name, split, cam_threshold_list,
                 iou_threshold_list, mask_root, multi_contour_eval, num_heads, _RESIZE_LENGTH=224):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.multi_contour_eval = multi_contour_eval
        self.num_heads = num_heads

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = kwargs['_RESIZE_LENGTH']
        self.cnt = 0
        self.num_correct = {head:
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
            for head in range(self.num_heads+1)
        }
        self.num_correct_top1 = {head:
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
            for head in range(self.num_heads+1)
        }
        self.num_correct_top5 = {head:
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
            for head in range(self.num_heads+1)
        }
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

        self.img_best_iou_lst = []

        self.cls_wrong_top1_loc_cls = []
        self.cls_wrong_top1_loc_mins = []
        self.cls_wrong_top1_loc_part = []
        self.cls_wrong_top1_loc_more = []
        self.cls_wrong_top1_loc_wrong = []

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremaps, image_id):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        sliced_multiple_iou = []
        for indx, scoremap in enumerate(scoremaps):
            sliced_multiple_iou.append([])
            boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
                scoremap=scoremap,
                scoremap_threshold_list=self.cam_threshold_list,
                multi_contour_eval=self.multi_contour_eval)

            boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

            multiple_iou = calculate_multiple_iou(
                np.array(boxes_at_thresholds),
                np.array(self.gt_bboxes[image_id]))

            idx = 0

            for nr_box in number_of_box_list:
                sliced_multiple_iou[indx].append(
                    max(multiple_iou.max(1)[idx:idx + nr_box]))
                idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = []
            best_head_index = 0
            for indx, _sliced_multiple_iou in enumerate(sliced_multiple_iou):
                _correct_threshold_indices = np.where(np.asarray(_sliced_multiple_iou) >= (_THRESHOLD / 100))[0]
                correct_threshold_indices.append(_correct_threshold_indices)
                self.num_correct[indx][_THRESHOLD][_correct_threshold_indices] += 1
                if len(correct_threshold_indices[indx]) > len(correct_threshold_indices[best_head_index]):
                    best_head_index = indx
            self.num_correct[self.num_heads][_THRESHOLD][correct_threshold_indices[best_head_index]] += 1
        self.cnt += 1

    def accumulate_with_all_matrices(self, scoremaps, image_id, pred_classes, target):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        sliced_multiple_iou = []
        best_iou = -1; best_bbox = []; best_map = []
        for indx, scoremap in enumerate(scoremaps):
            assert len(scoremaps) == 1, 'Number of Map should be equal to one for calculating all matrices'
            sliced_multiple_iou.append([])
            boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
                scoremap=scoremap,
                scoremap_threshold_list=self.cam_threshold_list,
                multi_contour_eval=self.multi_contour_eval)

            boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

            multiple_iou = calculate_multiple_iou(
                np.array(boxes_at_thresholds),
                np.array(self.gt_bboxes[image_id]))

            # top_1_idx = np.argsort(-multiple_iou.reshape(-1))[0]
            # if multiple_iou[top_1_idx][0] > best_iou:
            #     best_iou = multiple_iou[top_1_idx][0]
            #     best_bbox = boxes_at_thresholds[top_1_idx]
            #     best_map = scoremap
            for ind in range(multiple_iou.shape[1]):
                _multiple_iou = multiple_iou[:, ind:ind+1]
                top_1_idx = np.argsort(-_multiple_iou.reshape(-1))[0]
                if _multiple_iou[top_1_idx][0] > best_iou:
                    best_iou = _multiple_iou[top_1_idx][0]
                    best_bbox = boxes_at_thresholds[top_1_idx]
                    best_map = scoremap
                    max_gt_id = ind
                    # best_gt = self.gt_bboxes[image_id][ind:ind+1]
            cls_wrong, multi_instances, region_part, region_more, region_wrong = get_badcase_detail(pred_classes[0].item(), best_bbox, np.asarray(self.gt_bboxes[image_id]), target.item(), best_iou, max_gt_id)
            self.cls_wrong_top1_loc_cls.append(cls_wrong)
            self.cls_wrong_top1_loc_mins.append(multi_instances)
            self.cls_wrong_top1_loc_part.append(region_part)
            self.cls_wrong_top1_loc_more.append(region_more)
            self.cls_wrong_top1_loc_wrong.append(region_wrong)
            # get_badcase_detail([pred_classes[0], best_bbox], self.gt_bboxes[image_id], target, best_iou, max_gt_id)

            idx = 0

            for nr_box in number_of_box_list:
                sliced_multiple_iou[indx].append(
                    max(multiple_iou.max(1)[idx:idx + nr_box]))
                idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = []
            best_head_index = 0
            for indx, _sliced_multiple_iou in enumerate(sliced_multiple_iou):
                _correct_threshold_indices = np.where(np.asarray(_sliced_multiple_iou) >= (_THRESHOLD / 100))[0]
                correct_threshold_indices.append(_correct_threshold_indices)
                self.num_correct[indx][_THRESHOLD][_correct_threshold_indices] += 1

                if target == pred_classes[0]:
                    self.num_correct_top1[indx][_THRESHOLD][_correct_threshold_indices] += 1

                if target in pred_classes:
                    self.num_correct_top5[indx][_THRESHOLD][_correct_threshold_indices] += 1

                if len(correct_threshold_indices[indx]) > len(correct_threshold_indices[best_head_index]):
                    best_head_index = indx
            self.num_correct[self.num_heads][_THRESHOLD][correct_threshold_indices[best_head_index]] += 1
            if target == pred_classes[0]:
                self.num_correct_top1[self.num_heads][_THRESHOLD][correct_threshold_indices[best_head_index]] += 1

            if target in pred_classes:
                self.num_correct_top5[self.num_heads][_THRESHOLD][correct_threshold_indices[best_head_index]] += 1
        self.cnt += 1

        # topk = (1, 5)

        # cls_loc_err(topk_boxes, gt_label, gt_boxes, topk=topk, iou_th=0.5)
        
        return best_bbox#, best_gt

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        # max_box_acc = []
        max_box_acc = {head: [] for head in range(self.num_heads+1)}

        for _THRESHOLD in self.iou_threshold_list:
            for indx in range(self.num_heads+1):
                localization_accuracies = self.num_correct[indx][_THRESHOLD] * 100. / float(self.cnt)
                max_box_acc[indx].append(localization_accuracies.max())

        return max_box_acc

    def compute_on_all_matrcies(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        # max_box_acc = []
        max_box_acc = {head: [] for head in range(self.num_heads+1)}

        self.top1 = {head: [] for head in range(self.num_heads+1)}
        self.top5 = {head: [] for head in range(self.num_heads+1)}
        self.best_tau_list = {head: [] for head in range(self.num_heads+1)}

        self.curve_top_1_5 = {head: {
            'x': self.cam_threshold_list,
            'top1': dict(),
            'top5': dict()
        } for head in range(self.num_heads+1)}

        self.curve_s = {head: {
            'x': self.cam_threshold_list
        }  for head in range(self.num_heads+1)}

        localization_accuracies_at_each_th = {head: {} for head in range(self.num_heads+1)}
        for _THRESHOLD in self.iou_threshold_list:
            for indx in range(self.num_heads+1):
                localization_accuracies = self.num_correct[indx][_THRESHOLD] * 100. / float(self.cnt)
                localization_accuracies_at_each_th[indx][_THRESHOLD] = (localization_accuracies)
                max_box_acc[indx].append(localization_accuracies.max())

                self.curve_s[indx][_THRESHOLD] = localization_accuracies
                self.best_tau_list[indx].append(
                    self.cam_threshold_list[np.argmax(localization_accuracies)])

                loc_acc = self.num_correct_top1[indx][_THRESHOLD] * 100. / float(self.cnt)
                self.top1[indx].append(loc_acc.max())

                self.curve_top_1_5[indx]['top1'][_THRESHOLD] = deepcopy(loc_acc)

                loc_acc = self.num_correct_top5[indx][_THRESHOLD] * 100. / float(self.cnt)
                self.top5[indx].append(loc_acc.max())

                self.curve_top_1_5[indx]['top5'][_THRESHOLD] = deepcopy(loc_acc)


        return max_box_acc, localization_accuracies_at_each_th


def load_mask_image(file_path, resize_size):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask


def get_mask(mask_root, mask_paths, ignore_path, _RESIZE_LENGTH):
    """
    Ignore mask is set as the ignore box region setminus the ground truth
    foreground region.

    Args:
        mask_root: string.
        mask_paths: iterable of strings.
        ignore_path: string.

    Returns:
        mask: numpy.ndarray(size=(224, 224), dtype=np.uint8)
    """
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (_RESIZE_LENGTH, _RESIZE_LENGTH))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

    ignore_file = os.path.join(mask_root, ignore_path)
    ignore_box_mask = load_mask_image(ignore_file,
                                      (_RESIZE_LENGTH, _RESIZE_LENGTH))
    ignore_box_mask = ignore_box_mask > 0.5

    ignore_mask = np.logical_and(ignore_box_mask,
                                 np.logical_not(mask_all_instances))

    if np.logical_and(ignore_mask, mask_all_instances).any():
        raise RuntimeError("Ignore and foreground masks intersect.")

    return (mask_all_instances.astype(np.uint8) +
            255 * ignore_mask.astype(np.uint8))


# class MaskEvaluator(LocalizationEvaluator):
#     def __init__(self, **kwargs):
#         super(MaskEvaluator, self).__init__(**kwargs)

#         if self.dataset_name != "OpenImages":
#             raise ValueError("Mask evaluation must be performed on OpenImages.")

#         self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

#         # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
#         # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
#         self.num_bins = len(self.cam_threshold_list) + 2
#         self.threshold_list_right_edge = np.append(self.cam_threshold_list,
#                                                    [1.0, 2.0, 3.0])
#         self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float)
#         self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float)

#     def accumulate(self, scoremap, image_id):
#         """
#         Score histograms over the score map values at GT positive and negative
#         pixels are computed.

#         Args:
#             scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
#             image_id: string.
#         """
#         check_scoremap_validity(scoremap)
#         gt_mask = get_mask(self.mask_root,
#                            self.mask_paths[image_id],
#                            self.ignore_paths[image_id])

#         gt_true_scores = scoremap[gt_mask == 1]
#         gt_false_scores = scoremap[gt_mask == 0]

#         # histograms in ascending order
#         gt_true_hist, _ = np.histogram(gt_true_scores,
#                                        bins=self.threshold_list_right_edge)
#         self.gt_true_score_hist += gt_true_hist.astype(np.float)

#         gt_false_hist, _ = np.histogram(gt_false_scores,
#                                         bins=self.threshold_list_right_edge)
#         self.gt_false_score_hist += gt_false_hist.astype(np.float)

#     def compute(self):
#         """
#         Arrays are arranged in the following convention (bin edges):

#         gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
#         gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
#         tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

#         Returns:
#             auc: float. The area-under-curve of the precision-recall curve.
#                Also known as average precision (AP).
#         """
#         num_gt_true = self.gt_true_score_hist.sum()
#         tp = self.gt_true_score_hist[::-1].cumsum()
#         fn = num_gt_true - tp

#         num_gt_false = self.gt_false_score_hist.sum()
#         fp = self.gt_false_score_hist[::-1].cumsum()
#         tn = num_gt_false - fp

#         if ((tp + fn) <= 0).all():
#             raise RuntimeError("No positive ground truth in the eval set.")
#         if ((tp + fp) <= 0).all():
#             raise RuntimeError("No positive prediction in the eval set.")

#         non_zero_indices = (tp + fp) != 0

#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)

#         auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
#         auc *= 100

#         print("Mask AUC on split {}: {}".format(self.split, auc))
#         return auc


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        self.resize_length = kwargs['_RESIZE_LENGTH']

        if self.dataset_name != "OpenImages":
            raise ValueError("Mask evaluation must be performed on OpenImages.")

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])
        self.gt_true_score_hist = {head:
                np.zeros(self.num_bins, dtype=np.float)
            for head in range(self.num_heads+1)
        }
        self.gt_false_score_hist = {head:
                np.zeros(self.num_bins, dtype=np.float)
            for head in range(self.num_heads+1)
        }

    def accumulate(self, scoremaps, image_id):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id],
                           self.resize_length)

        
        IndPxAP = []
        gt_true_hist_lst = []
        gt_false_hist_lst = []
        for indx, scoremap in enumerate(scoremaps):
            check_scoremap_validity(scoremap)
            
            gt_true_scores = scoremap[gt_mask == 1]
            gt_false_scores = scoremap[gt_mask == 0]

            # histograms in ascending order
            gt_true_hist, _ = np.histogram(gt_true_scores,
                                        bins=self.threshold_list_right_edge)
            self.gt_true_score_hist[indx] += gt_true_hist.astype(np.float64)
            gt_true_hist_lst.append(gt_true_hist.astype(np.float64))

            gt_false_hist, _ = np.histogram(gt_false_scores,
                                            bins=self.threshold_list_right_edge)
            self.gt_false_score_hist[indx] += gt_false_hist.astype(np.float64)
            gt_false_hist_lst.append(gt_false_hist.astype(np.float64))
            IndPxAP.append(self.computeIndPxAP(gt_true_hist.astype(np.float64), gt_false_hist.astype(np.float64)))

        best_index = np.argsort(-np.asarray(IndPxAP))[0]
        self.gt_true_score_hist[self.num_heads] += gt_true_hist_lst[best_index]
        self.gt_false_score_hist[self.num_heads] += gt_false_hist_lst[best_index]
        # self.IndPxAP = IndPxAP


    def computeIndPxAP(self, gt_true_score_hist, gt_false_score_hist):
        num_gt_true = gt_true_score_hist.sum()
        tp = gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = gt_false_score_hist.sum()
        fp = gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            return 0#raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            return 0#raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        return auc

    
    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        Pxp_auc = {head: [] for head in range(self.num_heads+1)}
        for indx in range(self.num_heads+1):
            num_gt_true = self.gt_true_score_hist[indx].sum()
            tp = self.gt_true_score_hist[indx][::-1].cumsum()
            fn = num_gt_true - tp

            num_gt_false = self.gt_false_score_hist[indx].sum()
            fp = self.gt_false_score_hist[indx][::-1].cumsum()
            tn = num_gt_false - fp

            if ((tp + fn) <= 0).all():
                raise RuntimeError("No positive ground truth in the eval set.")
            if ((tp + fp) <= 0).all():
                raise RuntimeError("No positive prediction in the eval set.")

            non_zero_indices = (tp + fp) != 0

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
            auc *= 100

            Pxp_auc[indx] = auc

        print("Mask AUC on split {}: {}".format(self.split, Pxp_auc))
        return Pxp_auc


def _get_cam_loader(image_ids, scoremap_path):
    return torchdata.DataLoader(
        CamDataset(scoremap_path, image_ids),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True)


def evaluate_wsol(scoremap_root, metadata_root, mask_root, dataset_name, split,
                  multi_contour_eval, multi_iou_eval, iou_threshold_list,
                  cam_curve_interval=.001):
    """
    Compute WSOL performances of predicted heatmaps against ground truth
    boxes (CUB, ILSVRC) or masks (OpenImages). For boxes, we compute the
    gt-known box accuracy (IoU>=0.5) at the optimal heatmap threshold.
    For masks, we compute the area-under-curve of the pixel-wise precision-
    recall curve.

    Args:
        scoremap_root: string. Score maps for each eval image are saved under
            the output_path, with the name corresponding to their image_ids.
            For example, the heatmap for the image "123/456.JPEG" is expected
            to be located at "{output_path}/123/456.npy".
            The heatmaps must be numpy arrays of type np.float, with 2
            dimensions corresponding to height and width. The height and width
            must be identical to those of the original image. The heatmap values
            must be in the [0, 1] range. The map must attain values 0.0 and 1.0.
            See check_scoremap_validity() in util.py for the exact requirements.
        metadata_root: string.
        mask_root: string.
        dataset_name: string. Supports [CUB, ILSVRC, and OpenImages].
        split: string. Supports [train, val, test].
        multi_contour_eval:  considering the best match between the set of all
            estimated boxes and the set of all ground truth boxes.
        multi_iou_eval: averaging the performance across various level of iou
            thresholds.
        iou_threshold_list: list. default: [30, 50, 70]
        cam_curve_interval: float. Default 0.001. At which threshold intervals
            will the heatmaps be evaluated?
    Returns:
        performance: float. For CUB and ILSVRC, maxboxacc is returned.
            For OpenImages, area-under-curve of the precision-recall curve
            is returned.
    """
    print("Loading and evaluating cams.")
    meta_path = os.path.join(metadata_root, dataset_name, split)
    metadata = configure_metadata(meta_path)
    image_ids = get_image_ids(metadata)
    cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

    evaluator = {"OpenImages": MaskEvaluator,
                 "CUB": BoxEvaluator,
                 "ILSVRC": BoxEvaluator
                 }[dataset_name](metadata=metadata,
                                 dataset_name=dataset_name,
                                 split=split,
                                 cam_threshold_list=cam_threshold_list,
                                 mask_root=ospj(mask_root, 'OpenImages'),
                                 multi_contour_eval=multi_contour_eval,
                                 iou_threshold_list=iou_threshold_list)

    cam_loader = _get_cam_loader(image_ids, scoremap_root)
    for cams, image_ids in cam_loader:
        for cam, image_id in zip(cams, image_ids):
            evaluator.accumulate(t2n(cam), image_id)
    performance = evaluator.compute()
    if multi_iou_eval or dataset_name == 'OpenImages':
        performance = np.average(performance)
    else:
        performance = performance[iou_threshold_list.index(50)]

    print('localization: {}'.format(performance))
    return performance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scoremap_root', type=str,
                        default='train_log/scoremaps/',
                        help="The root folder for score maps to be evaluated.")
    parser.add_argument('--metadata_root', type=str, default='metadata/',
                        help="Root folder of metadata.")
    parser.add_argument('--mask_root', type=str, default='dataset/',
                        help="Root folder of masks (OpenImages).")
    parser.add_argument('--dataset_name', type=str,
                        help="One of [CUB, ImageNet, OpenImages].")
    parser.add_argument('--split', type=str,
                        help="One of [val, test]. They correspond to "
                             "train-fullsup and test, respectively.")
    parser.add_argument('--cam_curve_interval', type=float, default=0.01,
                        help="At which threshold intervals will the score maps "
                             "be evaluated?.")
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])

    args = parser.parse_args()
    evaluate_wsol(scoremap_root=args.scoremap_root,
                  metadata_root=args.metadata_root,
                  mask_root=args.mask_root,
                  dataset_name=args.dataset_name,
                  split=args.split,
                  cam_curve_interval=args.cam_curve_interval,
                  multi_contour_eval=args.multi_contour_eval,
                  multi_iou_eval=args.multi_iou_eval,
                  iou_threshold_list=args.iou_threshold_list,)


if __name__ == "__main__":
    main()
