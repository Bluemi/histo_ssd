from typing import List

import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision


def update_mean_average_precision(
        mean_average_precision: MeanAveragePrecision, ground_truth_boxes: torch.Tensor, predictions: List[torch.Tensor]
):
    """
    Updates the mean average precision metric.

    :param mean_average_precision: The MeanAveragePrecision object to update.
    :param ground_truth_boxes: Batch of ground truth boxes with shape [BATCH_SIZE, NUM_BOXES, 5] each entry with data
                               (class_label, left, top, right, bottom). Entries with class_label == -1.0 will be sorted
                               out.
    :param predictions: Batch of predictions with shape [BATCH_SIZE, NUM_PREDICTIONS, 6] each entry with data
                       (class_label, confidence, left, top, right, bottom).
    """
    assert ground_truth_boxes.shape[0] == len(predictions)  # batch size should be equal

    target = []
    preds = []
    # debug(predictions.shape)
    for sample_ground_truth_boxes, sample_predictions in zip(ground_truth_boxes.cpu(), predictions):
        sample_predictions = sample_predictions.cpu()
        # ground truth
        valid_box_indices = sample_ground_truth_boxes[:, 0] != -1.0  # filter out invalid boxes
        target_example = {
            'boxes': sample_ground_truth_boxes[valid_box_indices, 1:],
            'labels': sample_ground_truth_boxes[valid_box_indices, 0],
        }
        target.append(target_example)

        # predictions
        prediction_example = {
            'boxes': sample_predictions[:, 2:],
            'scores': sample_predictions[:, 1],
            'labels': sample_predictions[:, 0],
        }
        preds.append(prediction_example)

    mean_average_precision.update(preds, target)


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor):
    # Because the class prediction results are on the final dimension, `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


