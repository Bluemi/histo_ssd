import math
from collections import OrderedDict
from typing import List, Tuple, Reversible, Optional

import torch
from torch import nn
import torch.nn.functional as functional
from torch.hub import load_state_dict_from_url

from utils.bounding_boxes import create_anchor_boxes, multibox_detection

RELU_INPLACE = False

VGG16_KEY_MAPPING = OrderedDict([
    ('backbone.blocks.0.0.weight', 'backbone.features.0.weight'),
    ('backbone.blocks.0.0.bias', 'backbone.features.0.bias'),
    ('backbone.blocks.0.2.weight', 'backbone.features.2.weight'),
    ('backbone.blocks.0.2.bias', 'backbone.features.2.bias'),
    ('backbone.blocks.0.5.weight', 'backbone.features.5.weight'),
    ('backbone.blocks.0.5.bias', 'backbone.features.5.bias'),
    ('backbone.blocks.0.7.weight', 'backbone.features.7.weight'),
    ('backbone.blocks.0.7.bias', 'backbone.features.7.bias'),
    ('backbone.blocks.0.10.weight', 'backbone.features.10.weight'),
    ('backbone.blocks.0.10.bias', 'backbone.features.10.bias'),
    ('backbone.blocks.0.12.weight', 'backbone.features.12.weight'),
    ('backbone.blocks.0.12.bias', 'backbone.features.12.bias'),
    ('backbone.blocks.0.14.weight', 'backbone.features.14.weight'),
    ('backbone.blocks.0.14.bias', 'backbone.features.14.bias'),
    ('backbone.blocks.0.17.weight', 'backbone.features.17.weight'),
    ('backbone.blocks.0.17.bias', 'backbone.features.17.bias'),
    ('backbone.blocks.0.19.weight', 'backbone.features.19.weight'),
    ('backbone.blocks.0.19.bias', 'backbone.features.19.bias'),
    ('backbone.blocks.0.21.weight', 'backbone.features.21.weight'),
    ('backbone.blocks.0.21.bias', 'backbone.features.21.bias'),
    ('backbone.blocks.1.1.weight', 'backbone.extra.0.1.weight'),
    ('backbone.blocks.1.1.bias', 'backbone.extra.0.1.bias'),
    ('backbone.blocks.1.3.weight', 'backbone.extra.0.3.weight'),
    ('backbone.blocks.1.3.bias', 'backbone.extra.0.3.bias'),
    ('backbone.blocks.1.5.weight', 'backbone.extra.0.5.weight'),
    ('backbone.blocks.1.5.bias', 'backbone.extra.0.5.bias'),
    ('backbone.blocks.1.7.1.weight', 'backbone.extra.0.7.1.weight'),
    ('backbone.blocks.1.7.1.bias', 'backbone.extra.0.7.1.bias'),
    ('backbone.blocks.1.7.3.weight', 'backbone.extra.0.7.3.weight'),
    ('backbone.blocks.1.7.3.bias', 'backbone.extra.0.7.3.bias'),
    ('backbone.blocks.2.0.weight', 'backbone.extra.1.0.weight'),
    ('backbone.blocks.2.0.bias', 'backbone.extra.1.0.bias'),
    ('backbone.blocks.2.2.weight', 'backbone.extra.1.2.weight'),
    ('backbone.blocks.2.2.bias', 'backbone.extra.1.2.bias'),
    ('backbone.blocks.3.0.weight', 'backbone.extra.2.0.weight'),
    ('backbone.blocks.3.0.bias', 'backbone.extra.2.0.bias'),
    ('backbone.blocks.3.2.weight', 'backbone.extra.2.2.weight'),
    ('backbone.blocks.3.2.bias', 'backbone.extra.2.2.bias'),
    ('backbone.blocks.4.0.weight', 'backbone.extra.3.0.weight'),
    ('backbone.blocks.4.0.bias', 'backbone.extra.3.0.bias'),
    ('backbone.blocks.4.2.weight', 'backbone.extra.3.2.weight'),
    ('backbone.blocks.4.2.bias', 'backbone.extra.3.2.bias'),
    ('backbone.blocks.5.0.weight', 'backbone.extra.4.0.weight'),
    ('backbone.blocks.5.0.bias', 'backbone.extra.4.0.bias'),
    ('backbone.blocks.5.2.weight', 'backbone.extra.4.2.weight'),
    ('backbone.blocks.5.2.bias', 'backbone.extra.4.2.bias'),
])


VGG16_EARLY_KEY_MAPPING = OrderedDict([
    ('backbone.blocks.0.0.weight', 'backbone.features.0.weight'),
    ('backbone.blocks.0.0.bias', 'backbone.features.0.bias'),
    ('backbone.blocks.0.2.weight', 'backbone.features.2.weight'),
    ('backbone.blocks.0.2.bias', 'backbone.features.2.bias'),
    ('backbone.blocks.0.5.weight', 'backbone.features.5.weight'),
    ('backbone.blocks.0.5.bias', 'backbone.features.5.bias'),
    ('backbone.blocks.0.7.weight', 'backbone.features.7.weight'),
    ('backbone.blocks.0.7.bias', 'backbone.features.7.bias'),
    ('backbone.blocks.0.10.weight', 'backbone.features.10.weight'),
    ('backbone.blocks.0.10.bias', 'backbone.features.10.bias'),
    ('backbone.blocks.0.12.weight', 'backbone.features.12.weight'),
    ('backbone.blocks.0.12.bias', 'backbone.features.12.bias'),
    ('backbone.blocks.0.14.weight', 'backbone.features.14.weight'),
    ('backbone.blocks.0.14.bias', 'backbone.features.14.bias'),
    ('backbone.blocks.1.1.weight', 'backbone.features.17.weight'),
    ('backbone.blocks.1.1.bias', 'backbone.features.17.bias'),
    ('backbone.blocks.1.3.weight', 'backbone.features.19.weight'),
    ('backbone.blocks.1.3.bias', 'backbone.features.19.bias'),
    ('backbone.blocks.1.5.weight', 'backbone.features.21.weight'),
    ('backbone.blocks.1.5.bias', 'backbone.features.21.bias'),
    ('backbone.blocks.2.1.weight', 'backbone.extra.0.1.weight'),
    ('backbone.blocks.2.1.bias', 'backbone.extra.0.1.bias'),
    ('backbone.blocks.2.3.weight', 'backbone.extra.0.3.weight'),
    ('backbone.blocks.2.3.bias', 'backbone.extra.0.3.bias'),
    ('backbone.blocks.2.5.weight', 'backbone.extra.0.5.weight'),
    ('backbone.blocks.2.5.bias', 'backbone.extra.0.5.bias'),
    ('backbone.blocks.2.7.1.weight', 'backbone.extra.0.7.1.weight'),
    ('backbone.blocks.2.7.1.bias', 'backbone.extra.0.7.1.bias'),
    ('backbone.blocks.2.7.3.weight', 'backbone.extra.0.7.3.weight'),
    ('backbone.blocks.2.7.3.bias', 'backbone.extra.0.7.3.bias'),
    ('backbone.blocks.3.0.weight', 'backbone.extra.1.0.weight'),
    ('backbone.blocks.3.0.bias', 'backbone.extra.1.0.bias'),
    ('backbone.blocks.3.2.weight', 'backbone.extra.1.2.weight'),
    ('backbone.blocks.3.2.bias', 'backbone.extra.1.2.bias'),
    ('backbone.blocks.4.0.weight', 'backbone.extra.2.0.weight'),
    ('backbone.blocks.4.0.bias', 'backbone.extra.2.0.bias'),
    ('backbone.blocks.4.2.weight', 'backbone.extra.2.2.weight'),
    ('backbone.blocks.4.2.bias', 'backbone.extra.2.2.bias'),
    ('backbone.blocks.5.0.weight', 'backbone.extra.3.0.weight'),
    ('backbone.blocks.5.0.bias', 'backbone.extra.3.0.bias'),
    ('backbone.blocks.5.2.weight', 'backbone.extra.3.2.weight'),
    ('backbone.blocks.5.2.bias', 'backbone.extra.3.2.bias'),
    ('backbone.blocks.6.0.weight', 'backbone.extra.4.0.weight'),
    ('backbone.blocks.6.0.bias', 'backbone.extra.4.0.bias'),
    ('backbone.blocks.6.2.weight', 'backbone.extra.4.2.weight'),
    ('backbone.blocks.6.2.bias', 'backbone.extra.4.2.bias'),
])


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


def box_predictor(num_inputs: int, num_anchors: int, center_points: bool) -> nn.Conv2d:
    """
    Creates a layer for bounding box offsets, that can be executed on a feature map of shape
    [batch_size, num_inputs, y, x] and produces a bounding box offset prediction of shape
    [batch_size, num_anchors * 4, y, x].

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#bounding-box-prediction-layer

    :param num_inputs: The number of input channels for this conv layer
    :param num_anchors: Number of anchor boxes per spatial position
    :param center_points: Whether to only predict center points
    """
    predict_size = 2 if center_points else 4
    return nn.Conv2d(num_inputs, num_anchors * predict_size, kernel_size=3, padding=1)


def flatten_pred(pred: torch.Tensor) -> torch.Tensor:
    """
    Transforms the input prediction of shape [BATCH_SIZE, NUM_PREDICTIONS, HEIGHT, WIDTH] to
    [BATCH_SIZE, HEIGHT, WIDTH, NUM_PREDICTIONS] and flattens it to [BATCH_SIZE, HEIGHT*WIDTH*NUM_PREDICTIONS].
    num_class_predictions = num_anchors_per_pixel * (num_classes + 1)
    num_bbox_predictions = num_anchors_per_pixel * 4

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#concatenating-predictions-for-multiple-scales

    :param pred: A prediction of shape [BATCH_SIZE, NUM_CLASS_PREDICTIONS, HEIGHT, WIDTH]
    """
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Takes a list of predictions with elements of shape [NUM_FEATURE_MAPS, NUM_PREDICTIONS, HEIGHT, WIDTH],
    transforms each element to shape [BATCH_SIZE, -1] and concatenates them.
    Results in shape [BATCH_SIZE, NUM_PREDICTIONS]

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
        blk.append(nn.ReLU(inplace=RELU_INPLACE))
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


#  --- backbone ---
class NoConvException(Exception):
    pass


def search_last_conv(layers: Reversible[nn.Module]) -> nn.Conv2d:
    for layer in reversed(layers):
        if isinstance(layer, nn.Conv2d):
            return layer
        elif isinstance(layer, nn.Sequential):
            try:
                layer_modules = list(layer.modules())
                return search_last_conv(layer_modules)
            except NoConvException:
                pass
    raise NoConvException('Unable to find conv layer.\n{}'.format('\n'.join(map(str, layers))))


class Backbone(nn.Module):
    def __init__(self, blocks: List, debug: bool = False):
        """
        Creates new backbone with the given blocks.

        :param blocks: The layers of the network
        :param debug: If True, prints shape information for each layer output
        """
        super().__init__()
        self.debug = debug

        # model
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        output = []
        for block in self.blocks:
            x = block(x)
            if self.debug:
                print('Sequential: {}'.format(x.shape))
            output.append(x)
        return output

    def get_out_channels(self) -> List[int]:
        """
        Calculates the number of out channels for each block in the network. The length of the list is the number of
        feature maps produced by this model.
        """
        out_channels = []
        last_num_channels = None
        for block in self.blocks:
            if isinstance(block, nn.Sequential):
                block_list = list(block)
                last_conv_layer = search_last_conv(block_list)
                out_channels.append(last_conv_layer.out_channels)
                last_num_channels = last_conv_layer.out_channels
            else:
                if last_num_channels is None:
                    raise NoConvException('Could not find out channels')
                out_channels.append(last_num_channels)
        return out_channels

    @staticmethod
    def vgg11(debug=False):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        ]
        return Backbone(blocks=[nn.Sequential(*layers)], debug=debug)

    @staticmethod
    def vgg16(debug=False):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=RELU_INPLACE),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        ]
        return Backbone(blocks=[nn.Sequential(*layers)], debug=debug)

    @staticmethod
    def ssd_vgg16(debug=False):
        blocks = [
            # features
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                # patching ceil_mode to get original paper shape
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
            ),
            # first extra layer, fc6 and fc7
            nn.Sequential(
                # first extra layer
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
                    # fc6, atrous
                    nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6)),
                    nn.ReLU(inplace=RELU_INPLACE),
                    # fc7
                    nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),
                    nn.ReLU(inplace=RELU_INPLACE),
                )
            ),
            # extra feature layer 1
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
            ),
            # extra feature layer 2
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
            ),
            # extra feature layer 3
            nn.Sequential(
              nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
              nn.ReLU(inplace=RELU_INPLACE),
              nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
              nn.ReLU(inplace=RELU_INPLACE),
            ),
            # extra feature layer 4
            nn.Sequential(
              nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
              nn.ReLU(inplace=RELU_INPLACE),
              nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
              nn.ReLU(inplace=RELU_INPLACE),
            )
        ]
        return Backbone(blocks=blocks, debug=debug)

    @staticmethod
    def ssd_vgg16_early(debug=False):
        blocks = [
            # features
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
            ),
            nn.Sequential(
                # NEW EXTRA BLOCK FROM FIRST BLOCK
                # patching ceil_mode to get original paper shape
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=RELU_INPLACE),
            ),

            # first extra layer, fc6 and fc7
            nn.Sequential(
                # first extra layer
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
                    # fc6, atrous
                    nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6)),
                    nn.ReLU(inplace=RELU_INPLACE),
                    # fc7
                    nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),
                    nn.ReLU(inplace=RELU_INPLACE),
                )
            ),
            # extra feature layer 1
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
            ),
            # extra feature layer 2
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
            ),
            # extra feature layer 3
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
            ),
            # extra feature layer 4
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(inplace=RELU_INPLACE),
            )
        ]
        return Backbone(blocks=blocks, debug=debug)

    @staticmethod
    def tiny_base_net(debug=False):
        """
        Taken from https://d2l.ai/chapter_computer-vision/ssd.html#base-network-block and modified
        """
        layers = []
        num_filters = [3, 16, 32, 64]
        for i in range(len(num_filters) - 1):
            layers.append(down_sample_block(num_filters[i], num_filters[i+1]))

        blocks = [
            nn.Sequential(*layers),
            down_sample_block(64, 128),
            down_sample_block(128, 128),
            down_sample_block(128, 128),
            nn.AdaptiveMaxPool2d((1, 1)),
        ]
        return Backbone(blocks, debug=debug)


def blk_forward(
        feature_map: torch.Tensor, sizes: List[float], ratios: List[float], cls_predictor: nn.Conv2d,
        bbox_predictor: nn.Conv2d
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Taken from: https://d2l.ai/chapter_computer-vision/ssd.html#the-complete-model

    :param feature_map: The tensor to forward propagate with shape [BATCH_SIZE, NUM_INPUT_CHANNELS, IN_HEIGHT, IN_WIDTH]
    :param sizes: The sizes of the anchor boxes
    :param ratios: The ratios of the anchor boxes
    :param cls_predictor: The class predictor to use for classification
    :param bbox_predictor: The bounding box predictor to use for prediction
    :return: A tuple containing (anchors, cls_preds, bbox_preds):
                 - anchors: The anchor boxes created from the given block with shape [1, NUM_ANCHORS, 4]
                 - cls_preds: The class predictions for the given block with shape
                              [BATCH_SIZE, NUM_ANCHORS * (NUM_CLASSES + 1), OUT_HEIGHT, OUT_WIDTH]
                 - bbox_preds: The prediction of bounding boxes with shape
                               [BATCH_SIZE, NUM_ANCHORS * 4, OUT_HEIGHT, OUT_WIDTH]
    """
    anchors = create_anchor_boxes(shape=feature_map.shape[-2:], scales=sizes, ratios=ratios, device=feature_map.device)
    cls_preds = cls_predictor(feature_map)
    bbox_preds = bbox_predictor(feature_map)
    return anchors, cls_preds, bbox_preds


class SSDModel(nn.Module):
    def __init__(
            self, num_classes: int, backbone_arch: str = 'tiny', min_anchor_size: float = 0.2,
            max_anchor_size: float = 0.9, debug: bool = False, center_points: bool = False
    ):
        """
        Creates a new SSD Model with the given backbone architecture.
        :param num_classes: The number of classes to predict with this model
        :param backbone_arch: The backbone architecture. One of ["tiny", "vgg16"]
        :param min_anchor_size: The minimum size of the anchor boxes
        :param max_anchor_size: The maximum size of the anchor boxes
        :param debug: Whether to print status information
        :param center_points: If set to True, only center point offsets are predicted. Width and height offsets are
                              ignored.
        """
        super(SSDModel, self).__init__()
        self.debug = debug
        class_predictors: List[nn.Conv2d] = []
        bbox_predictors = []

        self.num_classes = num_classes
        self.center_points = center_points

        # create backbone
        if backbone_arch == 'tiny':
            backbone = Backbone.tiny_base_net(debug)
        elif backbone_arch == 'vgg16':
            backbone = Backbone.ssd_vgg16(debug)
        elif backbone_arch == 'vgg16early':
            backbone = Backbone.ssd_vgg16_early(debug)
        else:
            raise ValueError('Unknown backbone architecture: \"{}\"'.format(backbone_arch))

        self.backbone = backbone

        self.sizes = self._define_sizes(
            len(backbone.get_out_channels()), smin=min_anchor_size, smax=max_anchor_size,
            multiple_sizes=not self.center_points
        )

        ratios = [[1.0, 2.0, 0.5]] * len(backbone.get_out_channels())
        if self.center_points:
            # collapse all ratios into one
            ratios = [[1.0]] * len(backbone.get_out_channels())
        self.ratios = ratios

        for feature_map_index, out_channels in enumerate(self.backbone.get_out_channels()):
            num_anchors = self._get_num_anchors(feature_map_index)
            class_predictors.append(class_predictor(out_channels, num_anchors, num_classes))
            bbox_predictors.append(box_predictor(out_channels, num_anchors, center_points))

        self.class_predictors = nn.ModuleList(class_predictors)
        self.bbox_predictors = nn.ModuleList(bbox_predictors)

    @staticmethod
    def from_state_dict(
        state_dict_path: str, num_classes: int, backbone_arch: str = 'tiny', min_anchor_size: float = 0.2,
        max_anchor_size: float = 0.9, freeze_pretrained: bool = False, debug: bool = False, center_points: bool = False
    ):
        """
        Creates a new SSD Model and loads the given state dict.

        :param state_dict_path: The path to load. If DOWNLOAD the model will be downloaded on the fly.
        :param num_classes: The number of classes to predict with this model
        :param backbone_arch: The backbone architecture. One of ["tiny", "vgg16", "vgg16early"]
        :param min_anchor_size: The minimum size of the anchor boxes
        :param max_anchor_size: The maximum size of the anchor boxes
        :param freeze_pretrained: If True, pretrained layers are frozen at the start.
        :param debug: Whether to print status information
        :param center_points: Whether to only predict center points
        """
        assert backbone_arch in ('vgg16', 'vgg16early'), 'can only use pretrained vgg16'
        model = SSDModel(
            num_classes=num_classes,
            backbone_arch=backbone_arch,
            min_anchor_size=min_anchor_size,
            max_anchor_size=max_anchor_size,
            debug=debug,
            center_points=center_points,
        )
        if state_dict_path == 'DOWNLOAD':
            model_url = "https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth"
            state_dict = load_state_dict_from_url(model_url, progress=True)
        else:
            state_dict = torch.load('../ssd300_vgg16_coco-b556d3b4.pth')

        # remove unused keys
        remove_keys = []
        for key in state_dict.keys():
            if key.startswith('head.'):
                remove_keys.append(key)
        for remove_key in remove_keys:
            del state_dict[remove_key]

        # rename keys
        new_state_dict = []
        if backbone_arch == 'vgg16':
            key_mapping = VGG16_KEY_MAPPING
        elif backbone_arch == 'vgg16early':
            key_mapping = VGG16_EARLY_KEY_MAPPING
        else:
            raise ValueError('loading not supported for: {}'.format(backbone_arch))

        for new_key, old_key in key_mapping.items():
            new_state_dict.append(
                (new_key, state_dict[old_key])
            )
        new_state_dict = OrderedDict(new_state_dict)

        _missing, _unexpected = model.load_state_dict(new_state_dict, strict=False)
        # print('missing:', missing)
        # print('unexpected:', unexpected)
        # for mis, unex, in zip(missing, unexpected[1:]):
        # print("('{}', '{}')".format(mis, unex))

        if freeze_pretrained:
            for layer_name, layer in model.named_parameters():
                if layer_name in new_state_dict.keys():
                    layer.requires_grad = False

        return model

    def _get_num_anchors(self, feature_map_index: int):
        return len(self.sizes[feature_map_index]) + len(self.ratios[feature_map_index]) - 1

    @staticmethod
    def _define_sizes(num_feature_maps, smin=0.2, smax=0.9, multiple_sizes=True) -> List[List[float]]:
        """
        See paper page 6.
        :param num_feature_maps: The number of sizes to calculate. Corresponds to the number of feature maps.
        :param smin: The minimal anchor size
        :param smax: The maximal anchor size
        :return: A List of lists with sizes for each feature map with shape [NUM_FEATURE_MAPS, 2]
        """
        def _get_size(smin_arg, smax_arg, k_arg, m):
            return smin_arg + ((smax_arg - smin_arg) / (m - 1)) * (k_arg - 1)
        sizes = []
        for i in range(num_feature_maps):
            k = i+1
            size1 = _get_size(smin, smax, k, num_feature_maps)
            local_sizes = [size1]
            if multiple_sizes:
                size2 = math.sqrt(size1 * _get_size(smin, smax, k+1, num_feature_maps))  # sk
                local_sizes.append(size2)
            sizes.append(local_sizes)
        return sizes

    def unfreeze(self):
        for layer in self.parameters():
            layer.requires_grad = True

    def freeze_backbone(self):
        """
        Freezes all backbone layers.
        """
        for layer in self.backbone.parameters():
            layer.requires_grad = False

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes a batch of images and returns a tuple with three elements:
        1. anchor boxes: A tensor with shape [1, NUM_ANCHORS, 4] in ltrb-format
        2. class predictions: A tensor with shape [BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES + 1]
        3. bbox predictions: A tensor with shape [BATCH_SIZE, NUM_ANCHORS * 4]

        Taken from https://d2l.ai/chapter_computer-vision/ssd.html#the-complete-model and modified.

        :param x: A tensor with shape [BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH]
        """
        anchors, cls_preds, bbox_preds = [], [], []
        outputs = self.backbone(x)
        for i, feature_map in enumerate(outputs):
            # workaround to make pycharm debugger happy
            cls_predictor = self.class_predictors[i]
            assert isinstance(cls_predictor, nn.Conv2d)
            bbox_pred = self.bbox_predictors[i]
            assert isinstance(bbox_pred, nn.Conv2d)

            anchor, cls_pred, bbox_pred = blk_forward(
                feature_map,
                self.sizes[i],
                self.ratios[i],
                cls_predictor,
                bbox_pred,
            )
            anchors.append(anchor)
            cls_preds.append(cls_pred)
            bbox_preds.append(bbox_pred)
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def predict(
        anchors, cls_preds, bbox_preds, nms_iou_threshold=0.5, pos_threshold=0.2,
        num_pred_limit: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Uses the given model to predict boxes for the given batch of images.

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#prediction

    :param anchors: A tensor with shape [BATCH_SIZE, NUM_ANCHORS, 4].
    :param cls_preds: A tensor with shape [BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES + 1].
    :param bbox_preds: A tensor with shape [BATCH_SIZE, NUM_ANCHORS * 4] or [BATCH_SIZE, NUM_ANCHORS * 2] for center
                       points.
    :param nms_iou_threshold: The threshold nms uses to identify overlapping boxes in non-maximum suppression.
                              The smaller the threshold, the fewer boxes are kept.
    :param pos_threshold: Remove predictions with confidence smaller the pos_threshold.
    :param num_pred_limit: If given limits the number of predictions per sample.

    :return: A list with BATCH_SIZE entries. Each entry is a torch.Tensor with shape (NUM_PREDICTIONS, 6).
             Each entry of these tensors consists of (class_label, confidence, left, top, right, bottom).
    """
    # anchors, cls_preds, bbox_preds = model(images.to(device))
    cls_probs = functional.softmax(cls_preds, dim=2).permute(0, 2, 1)
    return multibox_detection(
        cls_probs, bbox_preds, anchors, nms_iou_threshold=nms_iou_threshold, pos_threshold=pos_threshold,
        num_pred_limit=num_pred_limit,
    )
