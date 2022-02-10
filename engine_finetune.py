# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import paddle
import paddle.amp as amp
import paddle.nn as nn
import paddle.optimizer as optim

from util.data import Mixup

import util.misc as misc
import util.lr_sched as lr_sched


def clear_grad_(optimizer: optim.Optimizer):
    if isinstance(optimizer, paddle.fluid.optimizer.Optimizer):
        optimizer.clear_gradients()
    else:
        optimizer.clear_grad()


def train_one_epoch(model: nn.Layer, criterion: nn.Layer,
                    data_loader: Iterable, optimizer: optim.Optimizer,
                    epoch: int, loss_scaler,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    clear_grad_(optimizer)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(data_iter_step / len(data_loader) + epoch, args)
            optimizer.set_lr(lr)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp.auto_cast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                           update_grad=(data_iter_step + 1) % accum_iter == 0)
        scale = loss_scaler.state_dict()['scale']
        if (data_iter_step + 1) % accum_iter == 0:
            clear_grad_(optimizer)

        paddle.device.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.update({'loss': loss_value_reduce, 'lr': lr, 'norm': norm, 'scale': scale})
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model):
    criterion = nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        # compute output
        with amp.auto_cast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred == target.reshape([1, -1]).expand_as(pred)
    return [correct[:min(k, maxk)].reshape([-1]).cast('float32').sum(0) * 100. / batch_size for k in topk]
