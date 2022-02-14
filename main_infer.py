# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
from PIL import Image
from imagenet_classes import id2class
import paddle.vision.transforms as transforms
from util.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import paddle
import models_vit



def get_args_parser():
    parser = argparse.ArgumentParser('MAE inference for image classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to infer')
    parser.add_argument('--resume', default=None, type=str, metavar='MODEL',
                        help='Model checkpoint to infer')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--img_path', default='demo/ILSVRC2012_val_00017997.JPEG', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    return parser


def main(args):
    # -------------------------------------------
    # Model
    # -------------------------------------------
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        # drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    model.eval()

    checkpoint = paddle.load(args.resume)
    model.set_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % args.resume)

    # -------------------------------------------
    # Transforms
    # -------------------------------------------
    t = []
    t.append(transforms.Resize(256, interpolation='bicubic'))  # to maintain same ratio w.r.t. 224 images
    t.append(transforms.CenterCrop(224))
    t.append(transforms.ToTensor())
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)

    # -------------------------------------------
    # Load Image
    # -------------------------------------------
    img = Image.open(args.img_path)
    input_tensor = transform(img)

    # -------------------------------------------
    # Inference
    # -------------------------------------------
    output = paddle.nn.Softmax(-1)(model(input_tensor[None]))
    prob, pred = output.topk(5, 1, True, True)
    prob, pred = prob[0], pred[0]

    print('Inference results:')
    for i in range(5):
        print(f'{id2class[str(pred[i].item())]}: {prob[i].item()}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
