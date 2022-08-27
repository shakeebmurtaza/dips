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
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
import cv2
from scipy.stats import norm
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.animation as animation

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


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

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
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args, masterport='29502'):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = masterport
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v

#
import argparse
#Courtesy of https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def fill_bounding_box(gray):
    # des = cv2.bitwise_not(gray)
    des = gray
    contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)

    gray = cv2.bitwise_not(des)
    return cv2.bitwise_not(gray)

def draw_bbox(img, box1, color1=(255, 0, 0), line_size = 2):
    for i in range(len(box1)):
        cv2.rectangle(img, (box1[i,0], box1[i,1]), (box1[i,2], box1[i,3]), color1, line_size)
    return img

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

def normalize_img_0_255(img):
    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    return img

def normalize_tensor(D):
    B = D.shape[0]
    C = D.shape[1]
    D_ = D.view(B,C,-1) 
    D_max = D_.max(dim=2)[0].unsqueeze(2).unsqueeze(2) 
    D_norm = (D/D_max).view(*D.shape)
    return D_norm

def filter_list_using_index(lst, lst_indxs):
    out = []
    for i in lst_indxs:
        out.append(lst[i])
    return out

def blur_bg(img, blur_img, binary_regions_bboxes, min_size = 0): #search min_size .04, .05, .06, .07, .08, .09 
    assert min_size >= 0.0 and min_size <= 1.0, 'Invalid Minimum Size'
    mask_blur_imgs = []
    mask_blur_imgs_bbox = []
    heads_lst = []
    bbox_areas_lst = []
    is_area_succeded = []
    h_img, w_img = img.shape[0], img.shape[1]
    total_area = float(h_img*w_img)
    index_of_largest_area = -1
    largest_area = -1
    for head_idx, binary_region in enumerate(binary_regions_bboxes):
        
        for binary_region_bbox in binary_region:
            map = np.zeros([img.shape[0], img.shape[1]])
            box_fused_input = draw_bbox(map.astype('uint8'), binary_region_bbox.reshape(-1, 4).astype(np.int32))
            box_fused_input = fill_bounding_box(box_fused_input)
            box_fused_input = np.where(box_fused_input == 255, 1, 0)
            area = np.sum(box_fused_input)/total_area
            is_area_succeded.append(area > min_size)
            bbox_areas_lst.append(area)
            box_fused_input = np.expand_dims(box_fused_input, axis=2)
            box_fused_input = (((img*box_fused_input))+((1-box_fused_input)*blur_img))

            mask_blur_imgs_bbox.append(binary_region_bbox.reshape(-1, 4).astype(np.int32))
            mask_blur_imgs.append(box_fused_input)
            heads_lst.append(head_idx)
            #"mean_1_4" if head_idx == 0 else str(head_idx-1)

            if area > largest_area:
                largest_area = area
                index_of_largest_area = len(mask_blur_imgs)-1

    if any(is_area_succeded):
        selected_indexs = np.where(is_area_succeded)[0].tolist()
    else:
        selected_indexs = [index_of_largest_area]

    assert len(selected_indexs) > 0, "No bbox founded"

    mask_blur_imgs = filter_list_using_index(mask_blur_imgs, selected_indexs)
    mask_blur_imgs_bbox = filter_list_using_index(mask_blur_imgs_bbox, selected_indexs)
    heads_lst = filter_list_using_index(heads_lst, selected_indexs)
    bbox_areas_lst = filter_list_using_index(bbox_areas_lst, selected_indexs)

    # for index in selected_indexs:

    # mask_blur_imgs = mask_blur_imgs[selected_indexs]
    # mask_blur_imgs_bbox = mask_blur_imgs_bbox[selected_indexs]
    # heads_lst = heads_lst[selected_indexs]
    # bbox_areas_lst = bbox_areas_lst[selected_indexs]
        

    bbox_areas_lst = np.asarray(bbox_areas_lst)
    bbox_areas_lst_mean = np.mean(bbox_areas_lst)
    bbox_areas_lst_std = np.std(bbox_areas_lst)
    size_dist = norm.pdf(bbox_areas_lst, loc=bbox_areas_lst_mean, scale=bbox_areas_lst_std)

    return mask_blur_imgs, mask_blur_imgs_bbox, heads_lst, size_dist, int(any(is_area_succeded))

def blur_bg_ind_img(img, blur_img, binary_regions_bboxes, min_size = 0, head_idx=None): #search min_size .04, .05, .06, .07, .08, .09 
    assert min_size >= 0.0 and min_size <= 1.0, 'Invalid Minimum Size'
    mask_blur_imgs = []
    mask_blur_imgs_bbox = []
    heads_lst = []
    bbox_areas_lst = []
    is_area_succeded = []
    h_img, w_img = img.shape[0], img.shape[1]
    total_area = float(h_img*w_img)
    index_of_largest_area = -1
    largest_area = -1
    for binary_region_bbox in binary_regions_bboxes:
        map = np.zeros([img.shape[0], img.shape[1]])
        box_fused_input = draw_bbox(map.astype('uint8'), binary_region_bbox.reshape(-1, 4).astype(np.int32))
        box_fused_input = fill_bounding_box(box_fused_input)
        box_fused_input = np.where(box_fused_input == 255, 1, 0)
        area = np.sum(box_fused_input)/total_area
        is_area_succeded.append(area > min_size)
        bbox_areas_lst.append(area)
        box_fused_input = np.expand_dims(box_fused_input, axis=2)
        box_fused_input = (((img*box_fused_input))+((1-box_fused_input)*blur_img))

        mask_blur_imgs_bbox.append(binary_region_bbox.reshape(-1, 4).astype(np.int32))
        mask_blur_imgs.append(box_fused_input)
        heads_lst.append(head_idx)
        #"mean_1_4" if head_idx == 0 else str(head_idx-1)

        if area > largest_area:
            largest_area = area
            index_of_largest_area = len(mask_blur_imgs)-1

    if any(is_area_succeded):
        selected_indexs = np.where(is_area_succeded)[0].tolist()
    else:
        selected_indexs = [index_of_largest_area]

    assert len(selected_indexs) > 0, "No bbox founded"

    mask_blur_imgs = filter_list_using_index(mask_blur_imgs, selected_indexs)
    mask_blur_imgs_bbox = filter_list_using_index(mask_blur_imgs_bbox, selected_indexs)
    heads_lst = filter_list_using_index(heads_lst, selected_indexs)
    bbox_areas_lst = filter_list_using_index(bbox_areas_lst, selected_indexs)

    # for index in selected_indexs:

    # mask_blur_imgs = mask_blur_imgs[selected_indexs]
    # mask_blur_imgs_bbox = mask_blur_imgs_bbox[selected_indexs]
    # heads_lst = heads_lst[selected_indexs]
    # bbox_areas_lst = bbox_areas_lst[selected_indexs]
        

    bbox_areas_lst = np.asarray(bbox_areas_lst)
    bbox_areas_lst_mean = np.mean(bbox_areas_lst)
    bbox_areas_lst_std = np.std(bbox_areas_lst)
    size_dist = norm.pdf(bbox_areas_lst, loc=bbox_areas_lst_mean, scale=bbox_areas_lst_std)

    return mask_blur_imgs, mask_blur_imgs_bbox, heads_lst, size_dist, int(any(is_area_succeded))

def resize_bbox(bbox, orignal_images_shape, targetSize = 224):
    y_ = targetSize#orignal_images_shape[0]
    x_ = targetSize#orignal_images_shape[1]
    
    x_scale = orignal_images_shape[1] / y_
    y_scale = orignal_images_shape[0] / x_
    kx = x_scale
    ky = y_scale

    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]

    return [[kx * x_min, ky * y_min, kx * x_max, ky * y_max]]

    # # original frame as named values
    # (origLeft, origTop, origRight, origBottom) = (bbox[0], bbox[1], bbox[2], bbox[3])

    # x = int(np.round(origLeft * x_scale))
    # y = int(np.round(origTop * y_scale))
    # xmax = int(np.round(origRight * x_scale))
    # ymax = int(np.round(origBottom * y_scale))
    # return [[x, y, xmax, ymax]]
    # return [[kx * x, ky * y, kx * xmax, ky * ymax]]

def get_chpooed_images(mask_blur_imgs, mask_blur_indx):
    return [(mask_blur_imgs+[''])[slice(ix,iy)] for ix, iy in zip([0]+mask_blur_indx, mask_blur_indx+[-1])][:-1]

def get_maps_for_binary_regions(attentions, evaluation_type):
    if evaluation_type == "all_heads_and_mean":
        cams_normalized = normalize_tensor(torch.cat([torch.mean(attentions, axis=1).unsqueeze(1), attentions], axis=1))
    elif evaluation_type == "1_5_heads_and_mean":
        cams_normalized = normalize_tensor(torch.cat([torch.mean(attentions, axis=1).unsqueeze(1), attentions[:, :5, :, :]], axis=1))
    elif evaluation_type == "1_4_heads_and_mean":
        cams_normalized = normalize_tensor(torch.cat([torch.mean(attentions, axis=1).unsqueeze(1), attentions[:, :4, :, :]], axis=1))
    elif evaluation_type == "1_5_heads_and_mean_1_5":
        cams_normalized = normalize_tensor(torch.cat([torch.mean(attentions[:, :5, :, :], axis=1).unsqueeze(1), attentions[:, :5, :, :]], axis=1))
    elif evaluation_type == "1_4_heads_and_mean_1_4":
        cams_normalized = normalize_tensor(torch.cat([torch.mean(attentions[:, :4, :, :], axis=1).unsqueeze(1), attentions[:, :4, :, :]], axis=1))
    elif evaluation_type == "all_heads_and_mean_1_4":
        cams_normalized = normalize_tensor(torch.cat([torch.mean(attentions[:, :4, :, :], axis=1).unsqueeze(1), attentions], axis=1))

    return cams_normalized

def reformat_id(img_id):
    tmp = str(Path(img_id).with_suffix(''))
    return tmp.replace('/', '_')

class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs

def get_video_from_imgs(imgs, save_path):
    def gen_frame(i):
        return imgs[i]#np.random.rand(300, 300)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(gen_frame(0), cmap='gray', interpolation='nearest')
    im.set_clim([0, 1])
    fig.set_size_inches([5, 5])

    plt.tight_layout()

    def update_img(n):
        tmp = gen_frame(n)
        im.set_data(tmp)
        return im

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, 100, interval=30)
    writer = animation.writers['ffmpeg'](fps=1)

    ani.save(f'{save_path}.mp4', writer=writer, dpi=72)
    return ani

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def get_video_from_multi_imgs(imgs, save_path):
    def gen_frame(i):
        return imgs[0][i], imgs[1][i], imgs[2][i], imgs[3][i], imgs[4][i], imgs[5][i]#np.random.rand(300, 300)

    img0, img1, img2, img3, img4, img5 = gen_frame(0)
    fig = plt.figure()
    ax = fig.add_subplot(161)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('SEED OLD')
    im = ax.imshow(img0, cmap='gray', interpolation='nearest')
    im.set_clim([0, 1])

    ax1 = fig.add_subplot(162)
    ax1.set_aspect('equal')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('SEED TEMP*2')
    im1 = ax1.imshow(img1, cmap='gray', interpolation='nearest')
    im1.set_clim([0, 1])

    ax2 = fig.add_subplot(163)
    ax2.set_aspect('equal')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('SEED TEMP*5')
    im2 = ax2.imshow(img2, cmap='gray', interpolation='nearest')
    im2.set_clim([0, 1])

    ax3 = fig.add_subplot(164)
    ax3.set_aspect('equal')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title('SEED TEMP*10')
    im3 = ax3.imshow(img3, cmap='gray', interpolation='nearest')
    im3.set_clim([0, 1])

    ax4 = fig.add_subplot(165)
    ax4.set_aspect('equal')
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title('SEED TEMP*20')
    im4 = ax4.imshow(img4, cmap='gray', interpolation='nearest')
    im4.set_clim([0, 1])

    ax5 = fig.add_subplot(166)
    ax5.set_aspect('equal')
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax5.set_title('NEW')
    im5 = ax5.imshow(img5, cmap='gray', interpolation='nearest')
    im5.set_clim([0, 1])

    fig.set_size_inches([30, 5])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_img(n):
        tmp, tmp1, tmp2, tmp3, tmp4, tmp5 = gen_frame(n)
        im.set_data(tmp)
        im1.set_data(tmp1)
        im2.set_data(tmp2)
        im3.set_data(tmp3)
        im4.set_data(tmp4)
        im5.set_data(tmp5)
        return im, im1, im2, im3, im4, im5

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, 100, interval=30)
    writer = animation.writers['ffmpeg'](fps=1)

    ani.save(f'{save_path}.mp4', writer=writer, dpi=72)
    return ani

def get_video_from_two_imgs(imgs, save_path):
    def gen_frame(i):
        return imgs[0][i], imgs[1][i]

    img0, img1 = gen_frame(0)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('SEED OLD')
    im = ax.imshow(img0, cmap='gray', interpolation='nearest')
    im.set_clim([0, 1])

    ax1 = fig.add_subplot(122)
    ax1.set_aspect('equal')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('SEED TEMP*2')
    im1 = ax1.imshow(img1, cmap='gray', interpolation='nearest')
    im1.set_clim([0, 1])

    fig.set_size_inches([10, 5])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_img(n):
        tmp, tmp1 = gen_frame(n)
        im.set_data(tmp)
        im1.set_data(tmp1)
        return im, im1

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, 5, interval=30)
    writer = animation.writers['ffmpeg'](fps=1)

    ani.save(f'{save_path}.mp4', writer=writer, dpi=72)
    return ani