from typing import List, Tuple

import torch
from torch import nn

from utils.bounding_boxes import create_anchor_boxes


def class_predictor(num_inputs: int, num_anchors: int, num_classes: int) -> nn.Conv2d:
    """
    Creates a class prediction layer, that can be executed on a feature map of shape [batch_size, num_inputs, y, x] and
    produces a class prediction of shape [batch_size, num_anchors * (num_classes + 1), y, x].

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#class-prediction-layer

    :param num_inputs: Number of input channels for this conv layer
    :param num_anchors: Number of anchor boxes per spatial position
    :param num_classes: Number of classes to predict for each bounding box
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def box_predictor(num_inputs: int, num_anchors: int) -> nn.Conv2d:
    """
    Creates a layer for bounding box offsets, that can be executed on a feature map of shape
    [batch_size, num_inputs, y, x] and produces a bounding box offset prediction of shape
    [batch_size, num_anchors * 4, y, x].

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#bounding-box-prediction-layer

    :param num_inputs: The number of input channels for this conv layer
    :param num_anchors: Number of anchor boxes per spatial position
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def flatten_pred(pred: torch.Tensor) -> torch.Tensor:
    """
    Transforms the input prediction of shape [batch_size, num_class_predictions, height, width] to
    [batch_size, height, width, num_class_predictions] and flattens it to [batch_size, -1].

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#concatenating-predictions-for-multiple-scales

    :param pred: A prediction of shape [batch_size, num_class_predictions, height, width]
    """
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Takes a list of predictions with elements of shape [batch_size, num_class_predictions, height, width],
    transforms each element to shape [batch_size, -1] and concatenates them.

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#concatenating-predictions-for-multiple-scales

    :param preds: A list of predictions to concatenate
    """
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_block(in_channels, out_channels) -> nn.Sequential:
    """
    Creates a block that can be used to scale a feature map of shape [batch_size, in_channels, height, width] to shape
    [batch_size, out_channels, height//2, width//2].

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#downsampling-block

    :param in_channels: The number of input channels of this block
    :param out_channels: The number of output channels of this block
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net() -> nn.Sequential:
    """
    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#base-network-block
    """
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_block(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i) -> nn.Module:
    """
    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#the-complete-model
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_block(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_block(128, 128)
    return blk


def blk_forward(
        x: torch.Tensor, block: nn.Module, sizes: List[float], ratios: List[float], cls_predictor: nn.Conv2d,
        bbox_predictor: nn.Conv2d
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Taken from: https://d2l.ai/chapter_computer-vision/ssd.html#the-complete-model

    :param x: The tensor to forward propagate with shape [BATCH_SIZE, NUM_INPUT_CHANNELS, IN_HEIGHT, IN_WIDTH]
    :param block: The block to use for propagation. Should accept x as input
    :param sizes: The sizes of the anchor boxes
    :param ratios: The ratios of the anchor boxes
    :param cls_predictor: The class predictor to use for classification
    :param bbox_predictor: The bounding box predictor to use for prediction
    :return: A tuple containing (y, anchors, cls_preds, bbox_preds):
                 - y: The output of the executed block with shape
                      [BATCH_SIZE, NUM_OUTPUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH]
                 - anchors: The anchor boxes created from the given block
                 - cls_preds: The class predictions for the given block with shape
                              [BATCH_SIZE, NUM_ANCHORS * (NUM_CLASSES + 1), OUT_HEIGHT, OUT_WIDTH]
                 - bbox_preds: The prediction of bounding boxes with shape
                               [BATCH_SIZE, NUM_ANCHORS * 4, OUT_HEIGHT, OUT_WIDTH]
    TODO: check shapes of args and returns
    """
    y = block(x)
    anchors = create_anchor_boxes(shape=y.shape[-2:], scales=sizes, ratios=ratios, device=y.device)
    cls_preds = cls_predictor(y)
    bbox_preds = bbox_predictor(y)
    return y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, num_classes):
        super(TinySSD, self).__init__()
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1

        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', class_predictor(idx_to_in_channels[i], self.num_anchors, num_classes))
            setattr(self, f'bbox_{i}', box_predictor(idx_to_in_channels[i], self.num_anchors))

    def forward(self, x):
        anchors, cls_preds, bbox_preds = [], [], []
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            x, anchor, cls_pred, bbox_pred = blk_forward(
                x,
                getattr(self, f'blk_{i}'),
                self.sizes[i],
                self.ratios[i],
                getattr(self, f'cls_{i}'),
                getattr(self, f'bbox_{i}')
            )
            anchors.append(anchor)
            cls_preds.append(cls_pred)
            bbox_preds.append(bbox_pred)
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
