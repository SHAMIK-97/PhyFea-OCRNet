# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_relu(name: str) -> nn.Sequential:
    container = nn.Sequential()
    relu = nn.ReLU()
    container.add_module(f'{name}_relu', relu)

    return container


def _max_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container


def _avg_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container


class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """

    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        loss = self.loss(outputs, labels)
        return torch.unsqueeze(loss, 0), outputs


class PhysicsFormer(FullModel):

    def __init__(self,model, loss,num_classes, iterations=2, image_size=(1024, 1024), **kwargs):

        super(PhysicsFormer, self).__init__(model, loss)
        self.open = []
        self.dilate = []
        self.classes = num_classes
        self.relu = _get_relu('relu')
        self.maxpool_1 = _max_pool2D("maxpool_1")
        self.avgpool_1 = _avg_pool2D("avgpool_1")
        self.pad_1 = nn.ConstantPad2d(1, 1)
        self.T = iterations
        self.image_size = image_size

    def opening(self, x):
        relu = x
        for iteration in range(self.T):
            x1 = self.maxpool_1(x)
            x = torch.matmul(x1, relu)

        return x

    def _rectification(self, original, dilated):

        offset = torch.sub(dilated, original, alpha=1)
        offset_mean = torch.mean(offset, dim=2, keepdim=True)
        offset_diff = torch.sub(offset, offset_mean, alpha=1)
        offset_relu = self.relu(offset_diff)
        final_dilation = torch.add(original, offset_relu, alpha=1)
        return final_dilation

    def selective_dilation(self, x):
        relu = x
        for iteration in range(self.T):
            x1 = self.avgpool_1(x)
            x2 = torch.matmul(x1, relu)
            x = self._rectification(x, x2)
        return x

    def final_operation(self, original, mode='opening'):

        if mode == 'opening':
            final_concatenated_opened = torch.mul(original, -1)
            final_concatenated_opening = self.pad_1(original)
            operated = self.opening(final_concatenated_opening)
            operated_normalized = F.normalize(operated)
            x = torch.matmul(operated_normalized, final_concatenated_opened)
            subtracted = torch.sub(final_concatenated_opened, x, alpha=1)
        else:
            final_concatenated_dilation = self.pad_1(original)
            operated = self.selective_dilation(final_concatenated_dilation)
            operated_normalized = F.normalize(operated)
            x = torch.matmul(operated_normalized, final_concatenated_dilation)
            subtracted = torch.sub(final_concatenated_dilation, x, alpha=1)

        l1_norm = torch.norm(subtracted, p=1)
        return l1_norm

    def forward(self, inputs, labels, *args, **kwargs):

        loss, logits = super().forward(inputs, labels, *args, **kwargs)
        logits_upscaled = nn.functional.interpolate(
            logits,
            size=self.image_size,
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        upscaled_softmax = logits_upscaled.softmax(dim=1)
        tensor_list = []
        perm = permutations(range(self.num_classes), 2)
        for i in perm:
            concatenated_tensor = torch.cat(
                (upscaled_softmax[:, i[0]:i[0] + 1, ::], upscaled_softmax[:, i[1]:i[1] + 1, ::]), dim=1)
            logits_mean = torch.mean(concatenated_tensor, dim=1, keepdim=True)
            logits_sub = torch.sub(concatenated_tensor, logits_mean, alpha=1)
            concat_relu = self.relu(logits_sub)
            tensor_list.append(concat_relu)

        final_concatenated = torch.cat(tensor_list, dim=1)

        norm_opened = self.final_operation(final_concatenated)
        norm_dilated = self.final_operation(final_concatenated, mode='dilation')
        if len(self.open)<=1:
            self.open.append(norm_opened)

        if len(self.dilate)<=1:
            self.dilate.append(norm_dilated)

        final_norm = torch.abs(norm_opened - norm_dilated)
        final_loss = torch.add(final_norm,loss,alpha=1)
        return final_loss, logits


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def create_logger(cfg, cfg_name, phase='train'):
    #root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not os.path.exists(cfg.OUTPUT_DIR):
        #print('=> creating {}'.format(cfg.OUTPUT_DIR))
        os.mkdir(cfg.OUTPUT_DIR)

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = cfg.OUTPUT_DIR+'/'+'/'+dataset+'/'+ cfg_name


    #print('=> creating {}'.format(final_output_dir))
    try:

        if not os.path.exists(cfg.OUTPUT_DIR+'/'+'/'+dataset):
            os.mkdir(cfg.OUTPUT_DIR+'/'+'/'+dataset)
        if not os.path.exists(cfg.OUTPUT_DIR+'/'+'/'+dataset+'/'+cfg_name):
            os.mkdir(cfg.OUTPUT_DIR+'/'+'/'+dataset+'/'+cfg_name)

    except FileExistsError:

        logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger_out = logging.getLogger(__name__)
        logger_out.error('final_output direcotry exists')


    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir+'/'+log_file
    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = cfg.LOG_DIR+dataset+'/'+model+'/'+cfg_name+'_'+time_str
    #print('=> creating {}'.format(tensorboard_log_dir))
    try:

        if not os.path.exists(cfg.LOG_DIR+'/'+dataset):
            os.mkdir(cfg.LOG_DIR+'/'+dataset)
        if not os.path.exists(cfg.LOG_DIR+'/'+dataset+'/'+model):
            os.mkdir(cfg.LOG_DIR+'/'+dataset+'/'+model)
        if not os.path.exists(cfg.LOG_DIR+'/'+dataset+'/'+model+'/'+cfg_name+'_'+time_str):
            os.mkdir(cfg.LOG_DIR+'/'+dataset+'/'+model+'/'+cfg_name+'_'+time_str)

    except FileExistsError:
        logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger_log = logging.getLogger(__name__)
        logger_log.error('final_log direcotry exists')

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
