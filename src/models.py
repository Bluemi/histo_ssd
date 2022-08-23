from typing import List, Tuple, Reversible

import torch
from torch import nn
import torch.nn.functional as functional

from utils.bounding_boxes import create_anchor_boxes, multibox_detection


RELU_INPLACE = False


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
    def tiny_base_net(debug=False):
        """
        Taken from https://d2l.ai/chapter_computer-vision/ssd.html#base-network-block
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
    def __init__(self, num_classes, backbone_arch='tiny', debug=False):
        super(SSDModel, self).__init__()
        self.debug = debug
        # TODO: sizes correct for different models?
        if backbone_arch == 'tiny':
            self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]  # sizes from d2l
        elif backbone_arch == 'vgg16':
            self.sizes = [[0.07], [0.15], [0.33], [0.51], [0.69], [0.87], [1.05]]  # sizes from torchvision

        self.ratios = [[1, 2, 0.5]] * 6
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        class_predictors: List[nn.Conv2d] = []
        bbox_predictors = []

        self.num_classes = num_classes
        # idx_to_in_channels = [128, 128, 128, 128]

        # create backbone
        if backbone_arch == 'tiny':
            backbone = Backbone.tiny_base_net(debug)
        elif backbone_arch == 'vgg16':
            backbone = Backbone.ssd_vgg16(debug)
        else:
            raise ValueError('Unknown backbone architecture: \"{}\"'.format(backbone_arch))

        self.backbone = backbone

        for out_channels in self.backbone.get_out_channels():
            class_predictors.append(class_predictor(out_channels, self.num_anchors, num_classes))
            bbox_predictors.append(box_predictor(out_channels, self.num_anchors))

        self.class_predictors = nn.ModuleList(class_predictors)
        self.bbox_predictors = nn.ModuleList(bbox_predictors)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes a batch of images and returns a tuple with three elements:
        1. anchor boxes: A tensor with shape [NUM_ANCHORS, 4] in ltrb-format
        2. class predictions: A tensor with shape [BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES + 1]
        3. bbox predictions: A tensor with shape [BATCH_SIZE, NUM_ANCHORS * 4]

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


def predict(anchors, cls_preds, bbox_preds, confidence_threshold=0.0) -> List[torch.Tensor]:
    """
    Uses the given model to predict boxes for the given batch of images.

    Taken from https://d2l.ai/chapter_computer-vision/ssd.html#prediction

    :param confidence_threshold: Filter out predictions with lower confidence than confidence_threshold.

    :return: A list with BATCH_SIZE entries. Each entry is a torch.Tensor with shape (NUM_PREDICTIONS, 6).
             Each entry of these tensors consists of (class_label, confidence, left, top, right, bottom).
    """
    # anchors, cls_preds, bbox_preds = model(images.to(device))
    cls_probs = functional.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)

    # filter out background and low confidences
    result = []
    for batch_output in output:
        idx = [i for i, row in enumerate(batch_output) if row[0] != -1 and row[1] >= confidence_threshold]
        filtered_batch_output = batch_output[idx]
        result.append(filtered_batch_output)
    return result
