import torch
import torchvision
from matplotlib import pyplot as plt

from datasets.lizard_detection import imread
from models import SSDModel
from utils.funcs import draw_boxes, debug


MODEL = 'tiny'
# noinspection PyRedeclaration
MODEL = 'vgg16'
# noinspection PyRedeclaration
MODEL = 'vgg16early'

if MODEL == 'tiny':
    LEVEL_SIZES = [32, 16, 8, 4, 1]
    NUM_BOXES_PER_PIXEL = 4
    IMAGE_SIZE = 256
elif MODEL == 'vgg16':
    LEVEL_SIZES = [38, 19, 10, 5, 3, 1]
    NUM_BOXES_PER_PIXEL = 4
    IMAGE_SIZE = 300
elif MODEL == 'vgg16early':
    LEVEL_SIZES = [75, 38, 19, 10, 5, 3, 1]  # TODO
    NUM_BOXES_PER_PIXEL = 4
    IMAGE_SIZE = 300

BATCH_SIZE = 1
VERBOSE = False


def main():
    # model = SSDModel(num_classes=1, debug=False, backbone_arch=MODEL, min_anchor_size=0.05, max_anchor_size=0.5)
    model = SSDModel.from_state_dict(
        state_dict_path='../ssd300_vgg16_coco-b556d3b4.pth', num_classes=1, debug=False, backbone_arch='vgg16early',
        min_anchor_size=0.05, max_anchor_size=0.5
    )

    model.eval()

    with torch.no_grad():
        image = torch.zeros((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))
        x = image
        for block in model.backbone.blocks:
            for layer in block:
                x = layer(x)
                if VERBOSE:
                    print('{}: {}'.format(layer, x.shape))
            if True:
                print('--> feature map with shape: {}'.format(x.shape))
        anchors, cls_preds, bbox_preds = model(image)

    anchors = anchors.squeeze(0)

    debug(anchors.shape)
    debug(cls_preds.shape)
    debug(bbox_preds.shape)
    debug(bbox_preds.reshape((BATCH_SIZE, -1, 4)).shape)

    # background_image = torch.zeros((1024, 1024, 3), dtype=torch.int)
    background_image = imread('res/pannuke_10_sample.png')

    add_boxes(background_image, anchors, x=15, y=16, level=0, num_boxes=1, color=(255, 0, 0))
    add_boxes(background_image, anchors, x=15, y=15, level=0, num_boxes=1, color=(0, 255, 0))
    # add_boxes(background_image, anchors, x=0.5, y=0.5, level=1, num_boxes=1, color=(0, 255, 0))
    # add_boxes(background_image, anchors, x=0.5, y=0.5, level=2, num_boxes=1, color=(0, 0, 255))

    plt.imshow(background_image)
    plt.show()


def test_torchvision():
    model = torchvision.models.detection.ssd.ssd300_vgg16(num_classes=1, pretrained=False, pretrained_backbone=False)
    print(model)


def add_boxes(image, anchors, x, y, level, color=(255, 0, 0), num_boxes=4):
    level_add = 0
    for i in range(level):
        level_add += LEVEL_SIZES[i] * LEVEL_SIZES[i]

    if isinstance(x, float):
        x = int(x * LEVEL_SIZES[level])
    if isinstance(y, float):
        y = int(y * LEVEL_SIZES[level])

    index = level_add + y * LEVEL_SIZES[level] + x

    draw_anchors = anchors[index * NUM_BOXES_PER_PIXEL:index * NUM_BOXES_PER_PIXEL + num_boxes]

    draw_boxes(image, draw_anchors, box_format='ltrb', color=color)


if __name__ == '__main__':
    main()
    # test_torchvision()
