"""
Defines functions to create and mutate bounding boxes.

TLBR-Format: Top-Left-Bottom-Right-Format: Bounding Boxes are saved as (top, left, bottom, right)
YXHW-Format: Y-X-Height-Width: Bounding Boxes are saved as (x-position, y-position, height, width)
"""
import torch
import math
from typing import List, Union, Tuple


def create_anchor_boxes(
        shape: Union[torch.Tensor, Tuple[int, int]], scales: List, ratios: List,
        device: Union[torch.device, str, None] = None
) -> torch.Tensor:
    """
    Creates anchor boxes centered on each point in shape. Anchor Boxes are in TLBR-format.

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
