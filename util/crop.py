# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import random
import numpy as np

import paddle.vision.transforms as transforms


class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    def _get_param(self, image, attempts=10):
        width, height = transforms.transforms._get_image_size(image)
        area = height * width

        target_area = np.random.uniform(*self.scale) * area
        log_ratio = tuple(math.log(x) for x in self.ratio)
        aspect_ratio = math.exp(np.random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        return i, j, h, w