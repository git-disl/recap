from __future__ import annotations

import os
from copy import copy

import cv2
from datasets import load_dataset
from torch.utils import data
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

skip_exec = True


def prepare_datasets(model_name: str, task_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    if task_name == 'glue':
        return prepare_datasets_glue(model_name, data_name, tokenizer, cache_dir, eval_key)
    elif task_name == 'img_class':
        if 'cifar' in data_name:
            return prepare_datasets_cifar(model_name, data_name, tokenizer, cache_dir, eval_key)
        elif data_name == 'tinyimagenet':
            return prepare_datasets_tinyimagenet(model_name, data_name, tokenizer, cache_dir, eval_key)
        else:
            raise NotImplementedError
    elif task_name == 'img_seg':
        if 'cityscapes' in data_name:
            return prepare_datasets_cityscapes(model_name, data_name, tokenizer, cache_dir, eval_key)
        elif 'kitti' in data_name:
            return prepare_datasets_kitti(model_name, data_name, tokenizer, cache_dir, eval_key)
        else:
            raise NotImplementedError


def prepare_datasets_glue(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    task_to_keys = {
        'cola': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'qqp': ('question1', 'question2'),
        'rte': ('sentence1', 'sentence2'),
        'sst2': ('sentence', None),
        'stsb': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }
    sentence1_key, sentence2_key = task_to_keys[data_name]

    # used to preprocess the raw data
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if 'label' in examples:
            result['labels'] = examples['label']
        return result

    raw_datasets = load_dataset('glue', data_name, cache_dir=cache_dir)

    if eval_key == 'val':
        for key in list(raw_datasets.keys()):
            if 'test' in key:
                raw_datasets.pop(key)

    column_names = raw_datasets['train'].column_names
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=column_names)

    if data_name == 'mnli':
        if eval_key == 'test':
            validation_datasets = {
                'test_matched': processed_datasets['validation_matched'],
                'test_mismatched': processed_datasets['validation_mismatched']
            }
        else:
            validation_datasets = {
                'validation_matched': processed_datasets['validation_matched'],
                'validation_mismatched': processed_datasets['validation_mismatched']
            }
    else:
        if eval_key == 'test':
            validation_datasets = {
                'test': processed_datasets['test']
            }
        else:
            validation_datasets = {
                'validation': processed_datasets['validation']
            }

    return processed_datasets['train'], validation_datasets, None


def prepare_datasets_cifar(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    train_ds, test_ds = load_dataset(data_name, cache_dir=cache_dir, split=['train', 'test'])
    # split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    image_mean, image_std = tokenizer.image_mean, tokenizer.image_std
    size = 224

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transform(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transform(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    train_ds.set_transform(train_transform)
    val_ds.set_transform(val_transform)
    test_ds.set_transform(val_transform)

    return train_ds, val_ds, test_ds


def prepare_datasets_tinyimagenet(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    train_ds, test_ds = load_dataset('Maysee/tiny-imagenet', cache_dir=cache_dir, split=['train', 'valid'])
    # split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    image_mean, image_std = tokenizer.image_mean, tokenizer.image_std
    size = 224

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transform(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transform(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    train_ds.set_transform(train_transform)
    val_ds.set_transform(val_transform)
    test_ds.set_transform(val_transform)

    return train_ds, val_ds, test_ds


def prepare_datasets_cityscapes(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    num_classes = 19

    train_tokenizer = copy(tokenizer)
    train_mul = 1
    train_tokenizer.size = {'height': int(512 * train_mul), 'width': int(1024 * train_mul)}
    eval_tokenizer = copy(tokenizer)
    eval_mul = 2
    eval_tokenizer.size = {'height': int(512 * eval_mul), 'width': int(1024 * eval_mul)}

    train_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/cityscapes/train.lst',
        tokenizer=train_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    val_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/cityscapes/val.lst',
        tokenizer=eval_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    test_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/cityscapes/test.lst',
        tokenizer=eval_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    return train_ds, val_ds, test_ds


def prepare_datasets_kitti(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    num_classes = 19

    train_tokenizer = copy(tokenizer)
    train_mul = 1
    train_tokenizer.size = {'height': int(375 * train_mul), 'width': int(1242 * train_mul)}
    eval_tokenizer = copy(tokenizer)
    eval_mul = 1
    eval_tokenizer.size = {'height': int(375 * eval_mul), 'width': int(1242 * eval_mul)}

    train_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/kitti/train.lst',
        tokenizer=train_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    val_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/kitti/val.lst',
        tokenizer=eval_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    return train_ds, val_ds, val_ds


class Cityscapes(data.Dataset):
    def __init__(self,
                 root,
                 list_path,
                 tokenizer,
                 num_classes=19,
                 ignore_label=255):

        super(Cityscapes, self).__init__()

        self.tokenizer = tokenizer
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.img_list = [line.strip().split() for line in open(root + list_path)]
        self.files = self.read_files()

        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

        self.target_mode = False
        self.image_mode = False

    def __len__(self):
        return len(self.files)

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        if 'cityscapes' in self.list_path:
            folder = 'cityscapes'
        else:
            folder = 'kitti'

        image = cv2.imread(os.path.join(self.root, folder, item["img"]), cv2.IMREAD_COLOR)

        if 'test' in self.list_path:
            return self.tokenizer(image)

        label = cv2.imread(os.path.join(self.root, folder, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)
        return self.tokenizer(image, label)
