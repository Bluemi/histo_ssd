from typing import List

import numpy as np
import torch


def mean_average_precision(
        ground_truth_boxes: List[torch.Tensor], predictions: List[torch.Tensor], intersection_over_union_thresholds: List[float],
        classes: List[str],
) -> float:
    """
    Calculates the mean average precision.

    :param ground_truth_boxes: The ground truth boxes of the dataset. Every entry in the list corresponds to an image
                               and has shape [NUM_BOXES, 5].
                               Each box has the 5 values (class_index, left, top, right, bottom).
    :param predictions: The model predictions to evaluate. Every entry in the list corresponds to an image and has
                        shape [NUM_PREDICTIONS, 7].
                        Each box has the 7 values (class_pred, class_score, left, top, right, bottom).
    :param intersection_over_union_thresholds: List of thresholds. Each threshold is used to match bounding boxes with
                                               greater iou than the threshold.
    :param classes: List of classes of the dataset.
    :return: The mean average precision of the predictions.
    """
    assert len(predictions) == len(ground_truth_boxes)

    map_values = []
    for intersection_over_union_threshold in intersection_over_union_thresholds:
        for class_index, class_name in enumerate(classes):
            map_value = map_for_class(ground_truth_boxes, predictions, class_index, intersection_over_union_threshold)
            map_values.append(map_value)
    return float(np.mean(map_values))


def map_for_class(
        ground_truth_boxes: List[torch.Tensor], predictions: List[torch.Tensor], class_index: int,
        intersection_over_union_threshold: float
) -> float:
    """
    Calculates average precision for given class and iou-threshold.

    :param ground_truth_boxes: The ground truth boxes of the dataset. Every entry in the list corresponds to an image
                               and has shape [NUM_BOXES, 5].
                               Each box has the 5 values (class_index, left, top, right, bottom).
    :param predictions: The model predictions to evaluate. Every entry in the list corresponds to an image and has
                        shape [NUM_PREDICTIONS, 7].
                        Each box has the 7 values (class_pred, class_score, left, top, right, bottom).
    :param class_index: The class index to evaluate for.
    :param intersection_over_union_threshold: A threshold used to match bounding boxes with greater iou than the
                                              threshold.
    :return: The average precision for the given class and iou-threshold
    """
