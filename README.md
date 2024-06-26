# CVPR24 Paper

Code for CVPR24 Paper - Resource-Efficient Transformer Pruning for Finetuning of Large Models

Fatih Ilhan, Gong Su, Selim Furkan Tekin, Tiansheng Huang, Sihao Hu, and Ling Liu, "Resource-Efficient Transformer Pruning for Finetuning of Large Models," IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, USA, Jun. 17-21, 2024.

## Setup

Python 3.10

Pytorch 2.0.1

Transformers 4.33

https://github.com/microsoft/nni

Please check requirements.txt for the list of other packages.

## Usage

CIFAR, TinyImageNet, GLUE datasets are automatically downloaded. You can download Cityscapes from https://www.cityscapes-dataset.com/ and KITTI from https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015.

### General usage for finetuning with RECAP:
```python main.py --task <TASK> --data <DATASET> --arch <MODEL> --init_sparse <r_p> --iter_sparse -<r_f> -num_pi <N_p> -num_pr <NUM_RECAP_ITERATIONS>```

### Example: Finetune ViT-base at CIFAR100 with 33% pruning and 87.5% masking in 10 iterations:
```python main.py --task img_class --data cifar100 --arch vit-base --init_sparse 0.33 --iter_sparse -0.875 -num_pi 2 -num_pr 10```

### Example: Finetune Mask2Former at Cityscapes with 50% pruning and 50% masking in 20 iterations:
```python main.py --task img_seg --data cityscapes --arch m2f --init_sparse 0.5 --iter_sparse -0.5 -num_pi 3 -num_pr 20```

### Example: Finetune BERT-base at CoLA with 17% pruning and 50% masking in 5 iterations:
```python main.py --task glue --data cola --arch bert-base-uncased --init_sparse 0.17 --iter_sparse -0.5 -num_pi 1 -num_pr 5```

### Parameters

All pruning/finetuning parameters are controlled from ``config.py``.
