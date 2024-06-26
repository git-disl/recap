import numpy as np
import torch
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names


def get_confusion_matrix(label, pred, size, num_class, ignore=255):
    """
    Calcute the confusion matrix by given label and pred
    """
    if pred.ndim == 4:
        output = pred.transpose(0, 2, 3, 1)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    else:
        seg_pred = pred
    seg_gt = np.asarray(label[:, :size[-2], :size[-1]], dtype=int)

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


def get_model_param_keys(model):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    keys = [[n for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            [n for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)]]

    return keys


def process_segmenter_output(outputs, target_sizes):
    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Scale back to preprocessed image size - (384, 384) for all models
    # masks_queries_logits = torch.nn.functional.interpolate(
    #     masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
    # )

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs).float()
    batch_size = class_queries_logits.shape[0]

    # Resize logits and compute semantic segmentation maps
    if target_sizes is not None:
        if batch_size != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

        semantic_segmentation = []
        for idx in range(batch_size):
            resized_logits = torch.nn.functional.interpolate(
                segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            )
            semantic_map = resized_logits[0].argmax(dim=0)
            semantic_segmentation.append(semantic_map)
    else:
        semantic_segmentation = segmentation.argmax(dim=1)
        semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

    return semantic_segmentation
