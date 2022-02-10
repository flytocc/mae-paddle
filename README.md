# PaddlePaddle复现："Masked Autoencoders Are Scalable Vision Learners"

## 1. 简介
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>

MAE是一种可扩展的计算机视觉自监督学习方法。在预训练阶段，对输入的图像随机遮挡一部分图像，并通过编码器和解码器实现重建。遮挡的部分对模型的骨干网络（即编码器）是不可见的，这可以大幅提升训练速度。这种自监督方法可以用于训练具有良好通用性的高容量，并且下游任务中的迁移性能优于有监督的预训练。

```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```
本仓库为基于 [官方pytorch实现](https://github.com/facebookresearch/mae) 的PaddlePaddle版本，模型与脚本与官方实现一致。

## 2. 复现精度
验收标准：ViT-B，Imagenet1k val 83.6%

复现精度：ViT-B，Imagenet1k val 83.454% (1400E结果，达到误差允许精度，1600E comming soon)

预训练模型在各个阶段的精度（通过1600E Pretrain的中间checkpoint进行Finetune）：

| Epoch | 800    | 1000   | 1100   | 1200   | 1300   | 1400   | 1600    |
| ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| TOP1  | 82.44% | 82.87% | 82.93% | 83.12% | 83.29% | 83.45% | Running |

- 预训练及1600Epochs Finetune在八卡环境完成，预训练单卡显存占用30G，Finetune单卡显存占用22G
- 除1600E外的Finetune在AIStudio完成，4卡V100 2天（对应AIStudio项目即将开源）
- 1400E Finetune 由于AIStudio运行的代码保存checkpoint逻辑有问题，为能保存精度最高的模型，在91epoch时中断转为本地训练

## 3. 数据集

ImageNet 1K

## 4. 环境依赖

- python 3.8.12
- paddlepaddle-gpu-2.2.1
- CUDA 11.5
- 其余依赖详见 `requirements.txt`

## 5. 快速开始

训练超参与官方完全一致。

### Pretrain

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_pretrain.py \
	--accum_iter 2 \
	--batch_size 256 \
	--model mae_vit_base_patch16 \
	--norm_pix_loss \
	--mask_ratio 0.75 \
	--epochs 1600 \
	--warmup_epochs 40 \
	--blr 1.5e-4 --weight_decay 0.05 \
	--data_path ${IMAGENET_DIR}
```

### Finetune

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_finetune.py \
  --accum_iter 1 \
  --batch_size 128 \
  --model vit_base_patch16 \
  --finetune ${PRETRAIN_CHKPT} \
  --epochs 100 \
  --blr 5e-4 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --data_path ${IMAGENET_DIR} --dist_eval
```

### Linprobe

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_linprobe.py \
  --accum_iter 2 \
  --batch_size 1024 \
  --model vit_base_patch16 --cls_token \
  --finetune ${PRETRAIN_CHKPT} \
  --epochs 90 \
  --blr 0.1 \
  --weight_decay 0.0 \
  --data_path ${IMAGENET_DIR} --dist_eval
```

### Evaluation

Finetune模型下载：[百度网盘](https://pan.baidu.com/s/1SqmQNhzCrbt6HtpRl4ozwA) 75on

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_finetune.py \
  --batch_size 128 \
  --model vit_base_patch16 \
  --resume ${FINETUNED_CHKPT} \
  --data_path ${IMAGENET_DIR} \
  --dist_eval --eval
```

## 6.代码结构

```
├── util # 功能性代码
├── engine_finetune.py  # Finetune核心代码
├── engine_pretrain.py  # 预训练核心代码
├── layer.py  # 网络结构
├── LICENSE
├── main_finetune.py  # Finetune脚本
├── main_linprobe.py  # LinearProbing脚本
├── main_pretrain.py  # 预训练脚本
├── models_mae.py  # mae预训练网络结构
├── models_vit.py  # ViT
├── README.md
├── requirements.txt
```

## 7. 模型信息

|      模型       |           权重            |       训练日志       |
| :------------: | :----------------------: | :-----------------: |
| Pretrain 1600E | pretrain_vit-b_1600e.pd  |     pretrain.log    |
| Finetune 1600E |         running          |       running       |
||
| Pretrain 1400E | pretrain_vit-b_1400e.pd  |       同1600E       |
| Finetune 1400E | finetuned_vit-b_1400e.pd | Finetuned_1400e.log |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1SqmQNhzCrbt6HtpRl4ozwA) （提取码：75on）

## 8. License

[CC-BY-NC 4.0 license](LICENSE)
