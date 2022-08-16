from typing import List, Tuple

import torch
from d2l.torch import d2l
from torch import nn
import torch.nn.functional as functional

from utils.bounding_boxes import create_anchor_boxes, multibox_detection


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


#  --- tiny base net ---

def tiny_base_net() -> nn.Sequential:
    """
    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#base-network-block
    """
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_block(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


#  --- vgg base net ---
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.conv_blocks = []
        for (num_convs, out_channels) in arch:
            self.conv_blocks.append(vgg_block(num_convs, out_channels))

    @staticmethod
    def default():
        return VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
            print('output shape:', x.shape)
        return x


def get_blk(i, base_net_arch='tiny') -> nn.Module:
    """
    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#the-complete-model
    """
    if i == 0:
        if base_net_arch == 'tiny':
            blk = tiny_base_net()
        else:
            raise ValueError('Unknown base_net_arch: \"{}\"'.format(base_net_arch))
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
    """
    y = block(x)
    anchors = create_anchor_boxes(shape=y.shape[-2:], scales=sizes, ratios=ratios, device=y.device)
    cls_preds = cls_predictor(y)
    bbox_preds = bbox_predictor(y)
    return y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, num_classes, base_net_arch='tiny', debug=False):
        super(TinySSD, self).__init__()
        self.debug = debug
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        self.blocks = []
        self.class_predictors = []
        self.bbox_predictors = []

        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            self.blocks.append(get_blk(i, base_net_arch=base_net_arch))
            self.class_predictors.append(class_predictor(idx_to_in_channels[i], self.num_anchors, num_classes))
            self.bbox_predictors.append(box_predictor(idx_to_in_channels[i], self.num_anchors))

    def forward(self, x):
        anchors, cls_preds, bbox_preds = [], [], []
        for i in range(5):
            x, anchor, cls_pred, bbox_pred = blk_forward(
                x,
                self.blocks[i],
                self.sizes[i],
                self.ratios[i],
                self.class_predictors[i],
                self.bbox_predictors[i],
            )
            if self.debug:
                print('blk output {} shape: {}'.format(i, x.shape))
            anchors.append(anchor)
            cls_preds.append(cls_pred)
            bbox_preds.append(bbox_pred)
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def predict(model, images, device='cpu', confidence_threshold=0.0) -> List[torch.Tensor]:
    """
    Uses the given model to predict boxes for the given batch of images.

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#prediction

    :param model: The model to use for prediction. model(images) should return
                  (anchor_boxes, class_predictions, and bounding_box_predictions).
    :param images: A batch of images with shape (BATCH_SIZE, DEPTH, HEIGHT, WIDTH).
    :param device: The torch device used for computation.
    :param confidence_threshold: Filter out predictions with lower confidence than confidence_threshold.

    :return: A list with BATCH_SIZE entries. Each entry is a torch.Tensor with shape (NUM_PREDICTIONS, 6).
             Each entry of these tensors consists of (class_label, confidence, left, top, right, bottom).
    """
    anchors, cls_preds, bbox_preds = model(images.to(device))
    cls_probs = functional.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)

    # filter out background and low confidences
    result = []
    for batch_output in output:
        idx = [i for i, row in enumerate(batch_output) if row[0] != -1 and row[1] >= confidence_threshold]
        filtered_batch_output = batch_output[idx]
        result.append(filtered_batch_output)
    return result
