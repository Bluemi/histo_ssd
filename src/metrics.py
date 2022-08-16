from typing import List

import torch
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
