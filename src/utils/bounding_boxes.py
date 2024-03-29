"""
Defines functions to create and mutate bounding boxes.

tlbr-Format:
    Top-Left-Bottom-Right-Format: Bounding Boxes are saved as (top, left, bottom, right). Each coordinate is normalized
    in respect to the image size.
yxhw-Format:
    Y-X-Height-Width: Bounding Boxes are saved as (x-position, y-position, height, width). Each coordinate is normalized
    in respect to the image size.
"""
import torch
from typing import List, Union, Tuple, Optional


def tlbr_to_yxhw(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of boxes from tlbr to yxhw format.

    Taken from https://d2l.ai/chapter_computer-vision/bounding-box.html and modified.

    :param boxes: The boxes to convert with shape (N, 4)
    """
    y = (boxes[:, 0] + boxes[:, 2]) / 2.0
    x = (boxes[:, 1] + boxes[:, 3]) / 2.0
    h = boxes[:, 2] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 1]
    return torch.stack((y, x, h, w), dim=1)


def yxhw_to_tlbr(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of boxes from yxhw to tlbr format.

    Taken from https://d2l.ai/chapter_computer-vision/bounding-box.html and modified.

    :param boxes: The boxes to convert with shape (N, 4)
    """
    t = boxes[:, 0] - boxes[:, 2] / 2.0
    l = boxes[:, 1] - boxes[:, 3] / 2.0
    b = boxes[:, 0] + boxes[:, 2] / 2.0
    r = boxes[:, 1] + boxes[:, 3] / 2.0
    return torch.stack((t, l, b, r), dim=1)


def box_centers(boxes: torch.Tensor) -> torch.Tensor:
    """
    Calculates the center points of the given boxes.

    :param boxes: A tensor with shape [NUM_BOXES, 4], containing (l, t, r, b) coordinates of the boxes.
    :return: A tensor with shape [NUM_BOXES, 2], containing (cx, xy).
    """
    return torch.stack(
        (
            (boxes[:, 0] + boxes[:, 2]) * 0.5,
            (boxes[:, 1] + boxes[:, 3]) * 0.5,
        ),
        dim=1
    )


def generate_random_boxes(
        num_boxes: int, min_size: float = 0.1, max_size: float = 1.0, device: str or None = None,
        seed: Optional[int] = None
) -> torch.Tensor:
    """
    Creates a batch of random bounding boxes with the shape (num_boxes, 4) in tlbr-format.

    :param num_boxes: The number of boxes to generate
    :param min_size: The minimal size for a bounding box
    :param max_size: The maximal size for a bounding box
    :param device: The device the bounding boxes are created on
    :param seed: Seed to use for randomness
    """
    if seed is not None:
        torch.random.manual_seed(seed)
    center = torch.rand((num_boxes, 2), device=device)
    height_width = torch.rand((num_boxes, 2), device=device) * (max_size - min_size) + min_size
    boxes = torch.stack((center, height_width), dim=1).reshape((num_boxes, 4))
    return yxhw_to_tlbr(boxes)


'''
def create_anchor_boxes(
        shape: Union[torch.Tensor, Tuple[int, int]], scales: List, ratios: List,
        device: Union[torch.device, str, None] = None
) -> torch.Tensor:
    """
    Creates anchor boxes centered on each point in shape. Anchor Boxes are in tlbr-format.
    
    Own implementation of create_anchor_boxes, but not used, because of worse model performance

    :param shape: One anchor box is created for every pixel in the given shape.
    :param scales: The scales for the anchor boxes
    :param ratios: The ratios for the anchor boxes
    """
    # create center points
    y_positions = torch.arange(0, shape[0], dtype=torch.float32, device=device) + 0.5  # + 0.5 to center on pixels
    x_positions = torch.arange(0, shape[1], dtype=torch.float32, device=device) + 0.5  # + 0.5 to center on pixels
    center_points = torch.dstack(torch.meshgrid(y_positions, x_positions, indexing='xy'))

    # normalize center points
    if shape[0] == shape[1]:
        center_points = center_points / shape[0]
    else:
        center_points[:, :, 0] /= float(shape[0])
        center_points[:, :, 1] /= float(shape[1])

    # create offsets
    offsets = []

    def _yx_offset_from_scale_ratio(s, r):
        return torch.tensor([s / math.sqrt(r) / 2.0, s * math.sqrt(r) / 2.0])

    for ratio in ratios:
        offsets.append(_yx_offset_from_scale_ratio(scales[0], ratio))
    for scale in scales[1:]:  # we skip the first scale, as it is already used in ratios
        offsets.append(_yx_offset_from_scale_ratio(scale, ratios[0]))

    offsets = torch.stack(offsets)
    if device:
        offsets = offsets.to(device)
    # create left upper points
    offsets = torch.stack([offsets, offsets], dim=1)
    offsets[:, 0, :] *= -1.0

    # add center points and offsets
    center_points = center_points.reshape((*shape, 1, 1, 2))
    offsets = offsets.reshape((offsets.shape[0], 2, 2))
    anchor_boxes = torch.add(center_points, offsets)
    return anchor_boxes.reshape(1, -1, 4)
'''


def create_anchor_boxes(
        shape: Union[torch.Tensor, Tuple[int, int]], scales: List, ratios: List,
        device: Union[torch.device, str, None] = None
) -> torch.Tensor:
    """
    Creates anchor boxes centered on each point in shape. Anchor Boxes are in ltrb-format.

    Taken from: https://d2l.ai/chapter_computer-vision/anchor.html#generating-multiple-anchor-boxes

    :param shape: One anchor box is created for every pixel in the given shape.
    :param scales: The scales for the anchor boxes
    :param ratios: The ratios for the anchor boxes
    :return: A list of anchor boxes with shape [1, NUM_ANCHORS, 4]
    """
    in_height, in_width = shape[0], shape[1]
    num_sizes, num_ratios = len(scales), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(scales, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w: torch.Tensor = torch.cat(
        (size_tensor * torch.sqrt(ratio_tensor[0]), scales[0] * torch.sqrt(ratio_tensor[1:]))
    ) * in_height / in_width  # Handle rectangular inputs
    h: torch.Tensor = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), scales[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Calculates the area of all given boxes.

    :param boxes: A tensor with shape (NUM_BOXES, 4), each sample containing (l, t, r, b).
    :return: The area of each box with shape (NUM_BOXES,)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection_over_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the intersection over union for a batch of bounding boxes in tlbr-format.

    Taken from https://d2l.ai/chapter_computer-vision/anchor.html#intersection-over-union-iou

    :param boxes1: First set of bounding boxes of shape (N, 4)
    :param boxes2: Second set of bounding boxes of shape (M, 4)
    :return: A tensor of shape (N,M) containing the intersection over unions. IoU[i][j] is the IoU of the i-th box of
             the first argument and the j-th box of the second argument.
    """
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upper_lefts`, `inter_lower_rights`, `inters`: (no. of boxes1, no. of boxes2, 2)
    inter_upper_lefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lower_rights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lower_rights - inter_upper_lefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_ground_truth_boxes(
        anchor_boxes: torch.Tensor, ground_truth: torch.Tensor, device='cpu', iou_threshold=0.5
) -> torch.Tensor:
    """
    Given is a batch of anchor boxes with shape (A, 4) and a batch of ground_truth boxes with shape (B, 4).
    Returns a mapping of shape (A,). The i-th entry in the result is the index of the assigned ground_truth box for
    the i-th anchor box. The given indices are between 0 <= index < B. The value -1 indicates, that this anchor box
    could not be assigned.

    Taken from https://d2l.ai/chapter_computer-vision/anchor.html#assigning-ground-truth-bounding-boxes-to-anchor-boxes

    :param anchor_boxes: A batch of anchor boxes with shape (A, 4)
    :param ground_truth: A batch of ground truth boxes with shape (B, 4)
    :param device: The device to compute on
    :param iou_threshold: Match boxes if they have iou > iou_threshold
    """
    num_anchors, num_gt_boxes = anchor_boxes.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor box i and the ground-truth bounding box j
    jaccard = intersection_over_union(anchor_boxes, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors: torch.Tensor, assigned_bb: torch.Tensor, eps=1e-6):
    """
    Transform for anchor box offsets.
    TODO: replace magic numbers with arguments

    Taken from https://d2l.ai/chapter_computer-vision/anchor.html#labeling-classes-and-offsets
    """
    c_anc = tlbr_to_yxhw(anchors)
    c_assigned_bb = tlbr_to_yxhw(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset


def offset_inverse(anchors: torch.Tensor, offset_preds: torch.Tensor) -> torch.Tensor:
    """
    Predict bounding boxes based on anchor boxes with predicted offsets.
    TODO: replace magic numbers with arguments

    Taken from https://d2l.ai/chapter_computer-vision/anchor.html#predicting-bounding-boxes-with-non-maximum-suppression

    :param anchors: Tensor with shape [NUM_ANCHORS, 4]
    :param offset_preds: Tensor with shape [NUM_ANCHORS, 4] or [NUM_ANCHORS, 2] for center points
    """
    center_points = offset_preds.shape[1] == 2
    anc = tlbr_to_yxhw(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    if center_points:
        pred_bbox_wh = anc[:, 2:]
    else:
        pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
    predicted_bbox = yxhw_to_tlbr(pred_bbox)
    return predicted_bbox


def multibox_target(
        anchors: torch.Tensor, labels: torch.Tensor, center_points, iou_match_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Label anchor boxes using ground-truth bounding boxes. Returns a tuple with three elements:
    1. The calculated offsets for assigned anchor boxes with shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4]
    2. A mask of shape [BATCH_SIZE, NUM_ANCHOR_BOXES*4], setting all negative examples to 0.0 and all positives
       examples to 1.0
    3. The class labels of the anchor boxes with shape [BATCH_SIZE, NUM_ANCHOR_BOXES]

    Taken from https://d2l.ai/chapter_computer-vision/anchor.html#labeling-classes-and-offsets and modified.

    :param anchors: List of anchor boxes with shape [1, NUM_ANCHOR_BOXES, 4] in tlbr-format.
    :param labels: Batch of ground truth boxes with shape [BATCH_SIZE, NUM_GT_BOXES, 5].
                   The 5 comes from (classlabel, t, l, b, r). If the classlabel is -1, the sample will be ignored.
    :param center_points: Whether the labels should be created for center points only or not
    :param iou_match_threshold: Match ground truth and anchor box, if iou > iou_match_threshold
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    bbox_size = 2 if center_points else 4

    for i in range(batch_size):
        label = labels[i, :, :]

        # filter out padding bounding boxes with class_label == -1
        class_label = label[:, 0]
        indices = torch.nonzero(class_label >= 0).flatten()
        label = label[indices]

        anchors_bbox_map = assign_anchor_to_ground_truth_boxes(
            anchors, label[:, 1:], device, iou_threshold=iou_match_threshold
        )

        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, bbox_size)
        # Initialize class labels and assigned bounding box coordinates with zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # Label classes of anchor boxes using their assigned ground-truth bounding boxes.
        # If an anchor box is not assigned any, we label its class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb)[:, :bbox_size] * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return bbox_offset, bbox_mask, class_labels


def non_maximum_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Calculate indices of boxes that do not overlap. Remove indices of boxes with lesser conf scores.
    Taken from https://d2l.ai/chapter_computer-vision/anchor.html#predicting-bounding-boxes-with-non-maximum-suppression

    :param boxes: The predicted boxes with shape (NUM_BOXES, 4)
    :param scores: The scores of the given boxes with shape (NUM_BOXES,)
    :param iou_threshold: Remove boxes that overlap more than the given iou threshold. The smaller the threshold,
                          the fewer boxes are kept.
    """
    score_indices = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while score_indices.numel() > 0:
        best_score_index = score_indices[0]
        keep.append(best_score_index)
        if score_indices.numel() == 1:
            break
        iou = intersection_over_union(
            boxes[best_score_index, :].reshape(-1, 4), boxes[score_indices[1:], :].reshape(-1, 4)
        ).reshape(-1)
        indices = torch.nonzero(iou <= iou_threshold).reshape(-1)
        score_indices = score_indices[indices + 1]
    return torch.tensor(keep, device=boxes.device, dtype=torch.int64)


def multibox_detection(
        cls_probs: torch.Tensor, offset_preds: torch.Tensor, anchors: torch.Tensor, nms_iou_threshold: float = 0.5,
        pos_threshold: float = 0.009999999, num_pred_limit: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Predict bounding boxes using non-maximum suppression.
    Taken from https://d2l.ai/chapter_computer-vision/anchor.html#predicting-bounding-boxes-with-non-maximum-suppression
    and modified.

    Returns a tensor of shape (BATCH_SIZE, NUM_ANCHOR_BOXES, 6).
    The 6 comes from (predicted class label, confidence, top, left, bottom, right) where "predicted class label" is -1
    for background. Bounding boxes without detection have -1 label class.

    :param cls_probs: The predicted class probabilities with shape (BATCH_SIZE, NUM_CLASSES, NUM_ANCHOR_BOXES)
    :param offset_preds: The predicted offsets with shape (BATCH_SIZE, NUM_ANCHORS*4) or (BATCH_SIZE, NUM_ANCHORS*2) for
                         center points.
    :param anchors: The anchor boxes which was predicted with shape (BATCH_SIZE, NUM_ANCHOR_BOXES, 4)
    :param nms_iou_threshold: The threshold nms uses to identify overlapping boxes in non-maximum suppression.
                              The smaller the threshold, the fewer boxes are kept.
    :param pos_threshold: A threshold uses for to low confidences.
    :param num_pred_limit: If given limits the number of predictions per sample.
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_anchors = anchors.shape[0]
    bbox_size = offset_preds.shape[1] // num_anchors
    out = []
    for i in range(batch_size):
        cls_prob = cls_probs[i]
        offset_pred = offset_preds[i].reshape(-1, bbox_size)
        conf, class_id = torch.max(cls_prob[1:], 0)  # ignore background probability

        # sort out all anchor box samples with conf <= pos_threshold
        pos_conf_indices = conf > pos_threshold
        if num_pred_limit is not None and torch.sum(pos_conf_indices) > num_pred_limit:
            sorted_confidence_indices = torch.argsort(conf, descending=True)
            pos_conf_indices = sorted_confidence_indices[:num_pred_limit]
        conf = conf[pos_conf_indices]
        class_id = class_id[pos_conf_indices]
        filtered_anchors = anchors[pos_conf_indices]
        offset_pred = offset_pred[pos_conf_indices]
        num_anchors = filtered_anchors.shape[0]

        predicted_bb = offset_inverse(filtered_anchors, offset_pred)
        keep = non_maximum_suppression(predicted_bb, conf, nms_iou_threshold)

        class_id = class_id[keep]
        conf = conf[keep]
        predicted_bb = predicted_bb[keep]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return out
