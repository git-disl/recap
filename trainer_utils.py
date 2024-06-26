from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_metric
from sklearn.metrics import accuracy_score
from torch.optim import *
from transformers import BertForSequenceClassification
from transformers import EvalPrediction
from transformers import ViTForImageClassification
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.training_args import TrainingArguments

import nni
from models.modeling_mask2former import Mask2FormerForUniversalSegmentation
from paths import get_path
from utils import get_model_param_keys

model_dispatcher = {
    'bert-base-uncased': BertForSequenceClassification,
    'bert-large-uncased': BertForSequenceClassification,
    'vit-base': ViTForImageClassification,
    'vit-large': ViTForImageClassification,
    'm2f': Mask2FormerForUniversalSegmentation
}


def build_model(pretrained_model_name_or_path: str, task_name: str, data_name: str, **kwargs):

    if data_name == 'cifar100':
        num_labels = 100
    elif data_name == 'tinyimagenet':
        num_labels = 200
    elif data_name == 'cityscapes' or data_name == 'kitti':
        num_labels = 19
    else:
        num_labels = 2

    if task_name == 'img_class':
        if 'vit' in pretrained_model_name_or_path:
            if pretrained_model_name_or_path == 'vit-base':
                model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                                  id2label=kwargs['id2label'],
                                                                  label2id=kwargs['label2id'], cache_dir='cache')
            elif pretrained_model_name_or_path == 'vit-large':
                model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224',
                                                                  id2label=kwargs['id2label'],
                                                                  label2id=kwargs['label2id'],
                                                                  ignore_mismatched_sizes=True, cache_dir='cache')
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif task_name == 'img_seg':
        if 'm2f' in pretrained_model_name_or_path:
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-IN21k-cityscapes-semantic", cache_dir='cache')
        else:
            raise NotImplementedError
    else:
        model = model_dispatcher[pretrained_model_name_or_path].from_pretrained(pretrained_model_name_or_path, num_labels=num_labels, cache_dir='cache')
    return model


def prepare_traced_trainer(model, args, data_content, training_params={}, for_train_flag=True, for_eval_flag=True,
                           tag='default', device=None, send_tag='train'):

    if 'img' in args.task:
        save_strategy = 'no' if 'prune' in tag else 'epoch'
        evaluation_strategy = 'no' if 'prune' in tag else 'epoch'
    else:
        save_strategy = 'no' if 'prune' in tag else 'epoch'
        evaluation_strategy = 'no' if 'prune' in tag else 'epoch'

    def compute_metrics(p: EvalPrediction):
        if args.task == 'glue':
            metric = load_metric('glue', args.data)

            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)

        elif args.task == 'img_class':
            predictions, labels = p.predictions, p.label_ids
            predictions = np.argmax(predictions, axis=1)
            result = dict(accuracy=accuracy_score(predictions, labels))

        elif args.task == 'img_seg':
            predictions, labels = p.predictions, p.label_ids
            predictions = predictions.sum(0)
            pos = predictions.sum(1)
            res = predictions.sum(0)
            tp = np.diag(predictions)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            score = IoU_array[pos + res - tp != 0].mean()
            result = dict(accuracy=score)
        else:
            raise NotImplementedError

        return result

    if tag == 'default':
        logging_dir = None
    else:
        logging_dir = get_path(args, 'TRAINER_FOLDER_DIR') + '/runs/' + tag

    if device is None:
        device = args.device

    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False

    if for_train_flag and for_eval_flag and args.task == 'img_seg':
        for_eval_flag = False

    num_steps = min(int(training_params.get('num_train_epochs', 3) * len(data_content['train']) / training_params.get('batch_size', 8)) + 5, 10000)

    training_args = TrainingArguments(output_dir=get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/{tag}',
                                      do_train=for_train_flag,
                                      do_eval=for_eval_flag,
                                      evaluation_strategy=evaluation_strategy,
                                      save_strategy=save_strategy,
                                      logging_strategy='epoch',
                                      logging_dir=logging_dir,
                                      logging_steps=500,
                                      per_device_train_batch_size=training_params.get('batch_size', 32),
                                      per_device_eval_batch_size=32,
                                      max_steps=num_steps,
                                      weight_decay=training_params.get('weight_decay', 1e-2),
                                      lr_scheduler_type='linear',
                                      dataloader_num_workers=1,
                                      learning_rate=training_params.get('learning_rate', 1e-4),
                                      save_total_limit=1,
                                      metric_for_best_model=args.metric_name,
                                      load_best_model_at_end=True,
                                      greater_is_better=True,
                                      disable_tqdm=True,
                                      optim='adamw_torch',
                                      seed=1024,
                                      use_mps_device=device == 'mps',
                                      no_cuda=no_cuda,
                                      remove_unused_columns=False)

    trainer = nni.trace(Trainer)(model=model,
                                 args=training_args,
                                 data_collator=data_content['collator'],
                                 train_dataset=data_content[send_tag],
                                 eval_dataset=data_content['val'],
                                 tokenizer=data_content['tokenizer'],
                                 compute_metrics=compute_metrics)

    return trainer


def predict(model_path, args, data_content, tag='default'):
    if not Path(model_path).exists():
        print(f'Model does not exist at {model_path}, exiting...')
        return {}

    if args.task == 'img_class' and tag == 'test':
        send_tag = 'test'
    else:
        send_tag = 'val'

    model = torch.load(model_path)
    trainer = prepare_traced_trainer(model.to(args.device), args, data_content, {}, for_train_flag=False, tag=tag)

    output = trainer.predict(data_content[send_tag], metric_key_prefix=tag)

    print(f'Metric: {output.metrics}')
    return output


def prepare_masked_trainer(args, trainer, max_steps, decay_zero=True):
    trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)

    if os.path.exists(get_path(args, 'ITER_MASKS_PATH')):
        masks = torch.load(get_path(args, 'ITER_MASKS_PATH'))
    else:
        masks = 1

    keys = get_model_param_keys(trainer.model)

    decay_parameters = get_parameter_names(trainer.model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    decay_val = 0 if decay_zero else trainer.args.weight_decay

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in trainer.model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": decay_val,
        },
        {
            "params": [
                p for n, p in trainer.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0,
        },
    ]
    _, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args)
    trainer.optimizer = CustomAdamW(keys, masks, optimizer_grouped_parameters, **optimizer_kwargs)


class CustomAdamW(AdamW):
    def __init__(self, keys, masks, args, **kwargs):
        super().__init__(args, **kwargs)
        self.keys = keys
        self.masks = masks

    def step(self, closure=None):
        c = -1
        for i in range(len(self.param_groups)):
            for j, param in enumerate(self.param_groups[i]['params']):
                c += 1
                key = self.keys[i][j]

                key_ = '.'.join(key.split('.')[:-1])
                _key = key.split('.')[-1]

                try:
                    if isinstance(self.masks, dict):
                        mask = self.masks[key_][_key]
                    else:
                        continue
                except:
                    continue

                if param.grad is None:
                    continue

                if mask.shape != param.grad.shape:
                    print(key)
                    raise RuntimeError

                param.grad *= mask.to(param.device)

        super(CustomAdamW, self).step(closure)
