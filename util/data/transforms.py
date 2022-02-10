import paddle
import paddle.vision.transforms as transforms
import paddle.vision.transforms.functional as F
from PIL import Image
import random
import numpy as np


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:

    def __init__(self, dtype=paddle.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return paddle.cast(paddle.to_tensor(np_img), dtype=self.dtype)


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = ('bilinear', 'bicubic')


class RandomResizedCropAndInterpolation(transforms.RandomResizedCrop):

    def _apply_image(self, img):
        interpolation = self.interpolation
        if self.interpolation == 'random':
            interpolation = random.choice(_RANDOM_INTERPOLATION)

        i, j, h, w = self._get_param(img)

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join(self.interpolation)
        else:
            interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
