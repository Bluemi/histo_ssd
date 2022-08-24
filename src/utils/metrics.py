from typing import List, Optional

import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision


EPSILON = 1e-7


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


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio: Optional[float] = None):
    """
    Calculates a loss value from class predictions and bounding box regression.

    Taken from: https://d2l.ai/chapter_computer-vision/ssd.html#defining-loss-and-evaluation-functions

    :param cls_preds: Class predictions of shape [BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES + 1]
    :param cls_labels: Class labels of shape [BATCH_SIZE, NUM_ANCHORS]
    :param bbox_preds: Bounding Box offset predictions of shape [BATCH_SIZE, NUM_ANCHORS * 4]
    :param bbox_labels: Bounding Box offsets with shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4]
    :param bbox_masks: A mask with shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4]. Each negative box has mask of (0, 0, 0, 0)
                       while each positive box has mask (1, 1, 1, 1).
    :param negative_ratio: If set enables hard negative mining. (negative_ratio * NUM_POSSIBLE_SAMPLES) negative samples
                           are used. If not set or set to None, all negative samples will be used.
    :return:
    """
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    batch_size, num_anchors, num_classes = cls_preds.shape
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1)
    if negative_ratio is not None:
        positive_mask = bbox_masks.reshape((batch_size, -1, 4))[:, :, 0]
        assert positive_mask.shape == torch.Size([batch_size, num_anchors])
        negative_mask = 1.0 - positive_mask

        num_positive_samples = torch.sum(positive_mask, dim=1)
        assert num_positive_samples.shape == torch.Size([batch_size])

        # num_negative_samples_per_batch = num_positive_samples_per_batch * negative_ratio
        num_negative_samples = num_positive_samples * negative_ratio
        num_samples = num_negative_samples + num_positive_samples
        num_negative_samples = num_negative_samples.to(torch.int)

        # sort higher class losses. By multiplying with negative mask, all positive samples are not considered for
        # negative sample choice
        loss_argsort = torch.argsort(cls*negative_mask, dim=1, descending=True)
        for i in range(batch_size):
            indices = loss_argsort[i, :num_negative_samples[i]]  # choose loss indices with the highest losses
            positive_mask[i][indices] = 1.0  # enable some negative samples
        cls = cls * positive_mask  # disable most of the negative samples
        # cls = torch.mean(cls, dim=1)  # use mean again, instead of sum / N
        # one could also try num_samples -> torch.mean(num_samples) to give all samples equal weight
        num_samples = torch.mean(num_samples)  # give all samples in batch equal weight
        cls = torch.sum(cls, dim=1) / torch.maximum(num_samples, torch.tensor(EPSILON))
    else:
        cls = torch.mean(cls, dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    assert cls.shape == torch.Size([batch_size])
    assert bbox.shape == torch.Size([batch_size])
    return cls + bbox


def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor):
    # Because the class prediction results are on the final dimension, `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
