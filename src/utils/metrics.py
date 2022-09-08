from typing import List, Optional, Tuple, Dict, Union

import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision

from utils.bounding_boxes import box_centers


EPSILON = 1e-7


def update_mean_average_precision(
        mean_average_precision: MeanAveragePrecision, ground_truth_boxes: torch.Tensor, predictions: List[torch.Tensor],
        divide_limit: int = 0
):
    """
    Updates the mean average precision metric.

    :param mean_average_precision: The MeanAveragePrecision object to update.
    :param ground_truth_boxes: Batch of ground truth boxes with shape [BATCH_SIZE, NUM_BOXES, 5] each entry with data
                               (class_label, left, top, right, bottom). Entries with class_label == -1.0 will be sorted
                               out.
    :param predictions: Batch of predictions with shape [BATCH_SIZE, NUM_PREDICTIONS, 6] each entry with data
                       (class_label, confidence, left, top, right, bottom).
    :param divide_limit: If set, divide predictions into smaller squares to fix det threshold
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

    # divide preds
    if divide_limit:
        divided_target = []
        divided_pred = []
        for t, p in zip(target, preds):
            ts, ps = divide_to_limit(t, p, limit=divide_limit)
            divided_target.extend(ts)
            divided_pred.extend(ps)
    else:
        divided_target = target
        divided_pred = preds

    mean_average_precision.update(divided_pred, divided_target)


def divide_to_limit(
        target: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], limit=100, outer_box=None
) -> Tuple[List, List]:
    """
    Divides the given targets and preds into smaller squares to limit the number of predictions per image to the given
    limit.

    :param target: The target dict with keys 'boxes' and 'labels'
    :param pred: The prediction dict with keys 'boxes', 'scores' and 'labels'
    :param limit: The limit to reduce to
    :param outer_box: The outer box for this division. If None the whole image is used.
    :return: Two lists (targets, preds) containing the same targets and preds but divided by position.
    """
    if pred['boxes'].shape[0] < limit:
        return [target], [pred]

    if outer_box is None:
        outer_box = torch.tensor([0.0, 0.0, 1.0, 1.0])

    # define outer boxes
    outer_box_center = torch.tensor([
        (outer_box[0] + outer_box[2]) * 0.5,
        (outer_box[1] + outer_box[3]) * 0.5
    ])
    new_preds = []
    new_targets = []

    for x in range(2):
        for y in range(2):
            new_outer_box = torch.clone(outer_box)
            new_outer_box[[x*2, y*2+1]] = outer_box_center

            # divide targets
            target_box_indices = box_indices_inside(target['boxes'], new_outer_box)
            new_target = {
                'boxes': target['boxes'][target_box_indices],
                'labels': target['labels'][target_box_indices],
            }

            # divide preds
            pred_box_indices = box_indices_inside(pred['boxes'], new_outer_box)
            new_pred = {
                'boxes': pred['boxes'][pred_box_indices],
                'scores': pred['scores'][pred_box_indices],
                'labels': pred['labels'][pred_box_indices],
            }

            ts, ps = divide_to_limit(new_target, new_pred, limit=limit, outer_box=new_outer_box)
            new_preds.extend(ps)
            new_targets.extend(ts)

    return new_targets, new_preds


def box_indices_inside(inner_boxes: torch.Tensor, outer_box: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor with shape [N,]. The i-th bool in the result is True, if the center of the i-th box in boxes is
    inside outer_box.

    :param inner_boxes: A tensor with shape [N, 4], each element is (l, t, r, b)
    :param outer_box: A tensor with shape [4,] with (l, t, r, b)
    """
    assert outer_box[0] < outer_box[2]
    assert outer_box[1] < outer_box[3]
    assert inner_boxes.shape[1] == 4

    centers = box_centers(inner_boxes)

    # check center - outer left
    # noinspection PyTypeChecker
    x_indices = torch.logical_and(centers[:, 0] >= outer_box[0], centers[:, 0] < outer_box[2])
    # noinspection PyTypeChecker
    y_indices = torch.logical_and(centers[:, 1] >= outer_box[1], centers[:, 1] < outer_box[3])

    return torch.logical_and(x_indices, y_indices)


def points_inside_boxes_indices(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor with shape (NUM_POINTS, NUM_BOXES). The entry (i, j) is True if the i-th point is inside the
    j-th box, otherwise False.
    :param points: Tensor with shape (NUM_POINTS, 2) with each entry (x, y).
    :param boxes: Tensor with shape (NUM_BOXES, 4) with each entry (l, t, r, b).
    """
    x_points: torch.Tensor = points[:, None, 0]
    x_indices = torch.logical_and(x_points >= boxes[None, :, 0], x_points <= boxes[None, :, 2])
    y_points: torch.Tensor = points[:, None, 1]
    y_indices = torch.logical_and(y_points >= boxes[None, :, 1], y_points <= boxes[None, :, 3])
    return torch.logical_and(x_indices, y_indices)


class ConfusionMatrix:
    def __init__(self):
        self.true_positives: int = 0
        self.false_positives: int = 0
        self.false_negatives: int = 0
        # true negatives are excluded as they don't make sense in object detection context

    def precision(self):
        if self.true_positives + self.false_positives == 0:
            return -1.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self):
        if self.true_positives + self.false_negatives == 0:
            return -1.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return -1.0
        return 2.0 * (p * r) / (p + r)

    def __str__(self):
        return 'ConfMat(tp={} fp={} fn={})'.format(self.true_positives, self.false_positives, self.false_negatives)


def update_confusion_matrix(
        confusion_matrix: ConfusionMatrix, ground_truth_boxes: torch.Tensor,
        predictions: Union[torch.Tensor, List[torch.Tensor]]
):
    """
    Updates the given confusion matrix
    :param confusion_matrix: The confusion matrix to update
    :param ground_truth_boxes: Batch of ground truth boxes with shape [BATCH_SIZE, NUM_BOXES, 5] each entry with data
                               (class_label, left, top, right, bottom). Entries with class_label == -1.0 will be sorted
                               out.
    :param predictions: Batch of predictions with shape [BATCH_SIZE, NUM_PREDICTIONS, 6] each entry with data
                       (class_label, confidence, left, top, right, bottom). The coordinates are only used to create
                       the center points
    :return:
    """
    assert ground_truth_boxes.shape[0] == len(predictions)  # batch size should be equal

    for sample_ground_truth_boxes, sample_predictions in zip(ground_truth_boxes.cpu(), predictions):
        sample_predictions = sample_predictions.cpu()
        pred_points = box_centers(sample_predictions[:, 2:])
        pred_labels = sample_predictions[:, 0]
        tp, fp, fn = calc_tp_fp_fn(sample_ground_truth_boxes, pred_labels, pred_points)
        confusion_matrix.true_positives += tp
        confusion_matrix.false_positives += fp
        confusion_matrix.false_negatives += fn


def calc_tp_fp_fn(ground_truth_boxes: torch.Tensor, pred_labels: torch.Tensor, pred_points) -> Tuple[int, int, int]:
    """
    Calculates the number of true positives, false positives and false negatives in the given ground truth boxes and
    predictions.
    Conditions for categories:
        true positive:
          - the prediction is in a gt box
          - no other point is closer to the gt box
          - the gt box has the same label
        false positive:
          - the prediction is not in a valid gt box
          - the closest containing gt box has a different labelx
          - another point is closer and contained in the gt box
        false negative:
          - gt box contains no prediction
          - all predictions have another gt box that is closer
    :param ground_truth_boxes: A tensor with shape [NUM_GT_BOXES, 5] with each entry
                               (class_label, left, top, right, bottom)
    :param pred_labels: A tensor with shape [NUM_PREDS] with class labels
    :param pred_points: A tensor with shape [NUM_PREDS, 2] containing the (x, y) coordinates of the predictions
    :return: A tuple containing the number of true positives, false positives and false negatives in the predictions
    """
    num_preds = pred_labels.shape[0]
    assert pred_points.shape[0] == num_preds

    # calculate gt center points for distance calculation
    gt_centers = box_centers(ground_truth_boxes[:, 1:])

    # calc inverse square distances between all boxes and all center points
    # distances should be a tensor with shape [NUM_PREDS, NUM_GT_BOXES]
    # calc: x_diff = (p1.x-p2.x)^2 and y_diff = (p1.y-p2.y)^2
    coordinate_diffs = torch.square(pred_points[:, None] - gt_centers[None, :])

    # calc: 1 / (x_diff + y_diff + 1)
    # add 1.0 for numeric stability. Exact value not important, as only the order counts
    inverse_distances = 1.0 / (coordinate_diffs[:, :, 0] + coordinate_diffs[:, :, 1] + 1.0)

    # set all distances, that are outside the box to -1
    outside_indices = torch.logical_not(points_inside_boxes_indices(pred_points, ground_truth_boxes[:, 1:]))
    inverse_distances[outside_indices] = -1.0

    max_inverse_distances, associated_gt_indices = torch.max(inverse_distances, dim=1)
    assert max_inverse_distances.shape[0] == num_preds
    associated_pred_mask = max_inverse_distances != -1  # true for preds that are in some gt box

    # count and remove preds with wrong label
    # get label for every pred of the associated gt box (this still contains labels for predictions that have no gt box)
    pred_true_labels = ground_truth_boxes[associated_gt_indices, 0]

    # true for every label, that is in a gt box with same label
    true_preds_mask = torch.logical_and(pred_true_labels == pred_labels, associated_pred_mask)
    false_preds_mask = torch.logical_not(true_preds_mask)

    associated_gt_indices[false_preds_mask] = -1.0  # set all gt indices to -1 that have wrong label

    # calc num_true_preds for every gt box
    unique_gt_indices, unique_pred_counts = torch.unique(associated_gt_indices, return_counts=True, sorted=True)

    fp_by_no_gt = 0  # we first assume 0. If unique_gt_indices[0] == -1, we will change this.
    if unique_gt_indices.shape[0] and unique_gt_indices[0] == -1:
        fp_by_no_gt = unique_pred_counts[0]  # number of predictions, that are associated with wrong or no gt box
        unique_pred_counts = unique_pred_counts[1:]  # remove fps with no gt
        unique_gt_indices = unique_gt_indices[1:]
    tp = len(unique_gt_indices)  # the number of gt boxes that are right predicted by at least one prediction

    fp_by_multi_det = torch.sum(unique_pred_counts - 1)  # number of preds already detected (-1 because one is right)
    fp = (fp_by_no_gt + fp_by_multi_det).item()

    # count number of gt boxes without prediction
    fn = len(ground_truth_boxes) - tp
    return tp, fp, fn


def calc_loss(
        cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio: Optional[float] = None,
        normalize_per: str = 'none', use_smooth_l1: bool = True,
) -> torch.Tensor:
    """
    Calculates a loss value from class predictions and bounding box regression.

    Taken from: https://d2l.ai/chapter_computer-vision/ssd.html#defining-loss-and-evaluation-functions

    :param cls_preds: Class predictions of shape [BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES + 1]
    :param cls_labels: Class labels of shape [BATCH_SIZE, NUM_ANCHORS]
    :param bbox_preds: Bounding Box offset predictions of shape [BATCH_SIZE, NUM_ANCHORS * 4] or
                       [BATCH_SIZE, NUM_ANCHORS * 2] if only center points are predicted.
    :param bbox_labels: Bounding Box offsets with shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4]
    :param bbox_masks: A mask with shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4] or [BATCH_SIZE, NUM_ANCHOR_BOXES*2] for center
                       points. Each negative box has mask of (0, 0, 0, 0) while each positive box has mask (1, 1, 1, 1).
    :param negative_ratio: If set enables hard negative mining. (negative_ratio * NUM_POSSIBLE_SAMPLES) negative samples
                           are used. If not set or set to None, all negative samples will be used.
    :param normalize_per: If set to "batch", hard negative samples are normalized per batch, if set to "sample" it is
                          normalized per sample, if set to "none" simply the mean is calculated.
    :param use_smooth_l1: Whether to use smoothed version of l1 loss
    :return: A single loss value
    """
    cls_loss, bbox_loss = calc_cls_bbox_loss(
        cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio, normalize_per, use_smooth_l1
    )
    return (cls_loss + bbox_loss).mean()


def calc_cls_bbox_loss(
        cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio: Optional[float] = None,
        normalize_per: str = 'none', use_smooth_l1: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates a loss value from class predictions and bounding box regression.

    Taken from: https://d2l.ai/chapter_computer-vision/ssd.html#defining-loss-and-evaluation-functions

    :param cls_preds: Class predictions of shape [BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES + 1]
    :param cls_labels: Class labels of shape [BATCH_SIZE, NUM_ANCHORS]
    :param bbox_preds: Bounding Box offset predictions of shape [BATCH_SIZE, NUM_ANCHORS * 4] or
                       [BATCH_SIZE, NUM_ANCHORS * 2] for center points.
    :param bbox_labels: Bounding Box offsets with shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4] or
                        [BATCH_SIZE, NUM_ANCHOR_BOXES*2].
    :param bbox_masks: A mask with shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4] or [BATCH_SIZE, NUM_ANCHOR_BOXES*2] for center
                       points. Each negative box has mask of (0, 0, 0, 0) while each positive box has mask (1, 1, 1, 1).
    :param negative_ratio: If set enables hard negative mining. (negative_ratio * NUM_POSSIBLE_SAMPLES) negative samples
                           are used. If not set or set to None, all negative samples will be used.
    :param normalize_per: If set to "batch", hard negative samples are normalized per batch, if set to "sample" it is
                          normalized per sample, if set to "none" simply the mean is calculated.
    :param use_smooth_l1: Whether to use smoothed version of l1 loss
    :return: A tuple [class_loss, bbox_loss] each with shape [BATCHSIZE].
    """
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    if use_smooth_l1:
        bbox_loss = nn.SmoothL1Loss(reduction='none')
    else:
        bbox_loss = nn.L1Loss(reduction='none')

    batch_size, num_anchors, num_classes = cls_preds.shape
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1)
    if negative_ratio is not None:
        positive_mask = bbox_masks.reshape((batch_size, num_anchors, -1))[:, :, 0]
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
        if normalize_per == 'batch':
            num_samples = torch.mean(num_samples)  # give all samples in batch equal weight
            cls = torch.sum(cls, dim=1) / torch.maximum(num_samples, torch.tensor(EPSILON))
        elif normalize_per == 'sample':
            cls = torch.sum(cls, dim=1) / torch.maximum(num_samples, torch.tensor(EPSILON))
        elif normalize_per == 'none':
            cls = torch.mean(cls, dim=1)
        else:
            raise ValueError('unknown normalize_per: "{}"'.format(normalize_per))
    else:
        cls = torch.mean(cls, dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    assert cls.shape == torch.Size([batch_size])
    assert bbox.shape == torch.Size([batch_size])
    return cls, bbox


def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor):
    # Because the class prediction results are on the final dimension, `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
