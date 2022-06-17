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
import math
from typing import List, Union, Tuple


def tlbr_to_yxhw(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of boxes from tlbr to yxhw format.

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

    :param boxes: The boxes to convert with shape (N, 4)
    """
    t = boxes[:, 0] - boxes[:, 2] / 2.0
    l = boxes[:, 1] - boxes[:, 3] / 2.0
    b = boxes[:, 0] + boxes[:, 2] / 2.0
    r = boxes[:, 1] + boxes[:, 3] / 2.0
    return torch.stack((t, l, b, r), dim=1)


def generate_random_boxes(num_boxes: int, device: str or None = None) -> torch.Tensor:
    """
    Creates a batch of random bounding boxes with the shape (num_boxes, 4) in tlbr-format.

    :param num_boxes: The number of boxes to generate
    """
    boxes = torch.rand((num_boxes, 2), device=device) / 2.0
    return torch.stack((boxes, boxes + torch.rand((num_boxes, 2), device=device) / 2.0), dim=1).reshape((num_boxes, 4))


def create_anchor_boxes(
        shape: Union[torch.Tensor, Tuple[int, int]], scales: List, ratios: List,
        device: Union[torch.device, str, None] = None
) -> torch.Tensor:
    """
    Creates anchor boxes centered on each point in shape. Anchor Boxes are in tlbr-format.

    :param shape: One anchor box is created for every pixel in the given shape.
    :param scales: The scales for the anchor boxes
    :param ratios: The ratios for the anchor boxes
    """
    # create center points
    y_positions = torch.arange(0, shape[0], dtype=torch.float32, device=device) + 0.5  # + 0.5 to center on pixels
    x_positions = torch.arange(0, shape[1], dtype=torch.float32, device=device) + 0.5  # + 0.5 to center on pixels
    center_points = torch.dstack(torch.meshgrid(y_positions, x_positions, indexing='ij'))

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
    return anchor_boxes


def intersection_over_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the intersection over union for a batch of bounding boxes in tlbr-format.

    :param boxes1: First set of bounding boxes of shape (N, 4)
    :param boxes2: Second set of bounding boxes of shape (M, 4)
    :return: A tensor of shape (N,M) containing the intersection over unions. IoU[i][j] is the IoU of the i-th box of
             the first argument and the j-th box of the second argument.
    """
    def _area_of_boxes(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def _area_of_intersections(tl, br):
        inters = br - tl
        return inters[:, :, 0] * inters[:, :, 1]

    # shape (N, M, 2)
    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # find out invalid intersection areas
    invalid_mask = 1.0 - torch.logical_or(
        torch.gt(top_left[:, :, 0], bottom_right[:, :, 0]),  # top > bottom -> invalid
        torch.gt(top_left[:, :, 1], bottom_right[:, :, 1])   # left > right -> invalid
    ).to(torch.float)

    boxes1_areas = _area_of_boxes(boxes1)
    boxes2_areas = _area_of_boxes(boxes2)
    intersection_areas = _area_of_intersections(top_left, bottom_right)

    return torch.abs(invalid_mask * intersection_areas / (boxes1_areas[:, None] + boxes2_areas - intersection_areas))
