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
from typing import Iterable

import paddle
import paddle.amp as amp
import paddle.optimizer as optim

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: paddle.nn.Layer,
                    data_loader: Iterable, optimizer: optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.clear_grad()

    for data_iter_step, (samples,) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(data_iter_step / len(data_loader) + epoch, args)
            optimizer.set_lr(lr)

        with amp.auto_cast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                           update_grad=(data_iter_step + 1) % accum_iter == 0)
        scale = loss_scaler.state_dict()['scale']
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.clear_grad()

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