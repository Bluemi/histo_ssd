import torch
from torchmetrics.detection import MeanAveragePrecision

from utils.funcs import debug


def update_mean_average_precision(
        mean_average_precision: MeanAveragePrecision, ground_truth_boxes: torch.Tensor, predictions: torch.Tensor,
        score_threshold: float = -1.0,
):
    """
    Updates the mean average precision metric.

    :param mean_average_precision: The MeanAveragePrecision object to update
    :param ground_truth_boxes: Batch of ground truth boxes with shape [BATCH_SIZE, NUM_BOXES, 5] each entry with data
                               (class_label, left, top, right, bottom)
    :param predictions: Batch of predictions with shape [BATCH_SIZE, NUM_PREDICTIONS, 6] each entry with data
                        (class_label, confidence, left, top, right, bottom).
    :param score_threshold: Class scores under this threshold are not considered for calculation. Defaults to -1.0,
                            which leads to all predictions being considered.
    """
    assert ground_truth_boxes.shape[0] == predictions.shape[0]  # batch size should be equal

    target = []
    preds = []
    for sample_ground_truth_boxes, sample_predictions in zip(ground_truth_boxes, predictions):
        target_example = {
            'boxes': sample_ground_truth_boxes[:, 1:],
            'labels': sample_ground_truth_boxes[:, 0],
        }
        target.append(target_example)
        debug(target_example['labels'])

        threshold_indices = sample_predictions[:, 1] >= score_threshold
        prediction_example = {
            'boxes': sample_predictions[threshold_indices, 2:],
            'scores': sample_predictions[threshold_indices, 1],
            'labels': sample_predictions[threshold_indices, 0],
        }
        debug(prediction_example['labels'])
        preds.append(prediction_example)

    mean_average_precision.update(preds, target)
