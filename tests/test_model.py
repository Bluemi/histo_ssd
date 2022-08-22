import torch
from matplotlib import pyplot as plt

from models import SSDModel, Backbone
from utils.funcs import draw_boxes
from torchvision.models.detection import ssd


MODEL = 'tiny'
# noinspection PyRedeclaration
MODEL = 'vgg16'

if MODEL == 'tiny':
    LEVEL_SIZES = [32, 16, 8, 4, 1]
    NUM_BOXES_PER_PIXEL = 4
    IMAGE_SIZE = 256
elif MODEL == 'vgg16':
    LEVEL_SIZES = [38, 19, 10, 5, 3, 1]
    NUM_BOXES_PER_PIXEL = 3
    IMAGE_SIZE = 300


def main():
    model = SSDModel(num_classes=1, debug=True, backbone_arch=MODEL)
    # for parameter in model.parameters():
    # print(parameter.data.shape)
    model.eval()

    # base_model: torch.nn.Sequential = model.blocks[0]
    # print(base_model._modules)

    with torch.no_grad():
        image = torch.zeros((7, 3, IMAGE_SIZE, IMAGE_SIZE))
        x = image
        for block in model.backbone.layers:
            for layer in block:
                x = layer(x)
                print('{}: {}'.format(layer, x.shape))
        anchors, cls_preds, bbox_preds = model(image)

    anchors = anchors.squeeze(0)

    # debug(anchors.shape)
    # debug(cls_preds.shape)
    # debug(bbox_preds.reshape((1, -1, 4)).shape)

    black_image = torch.zeros((1024, 1024, 3), dtype=torch.int)

    add_boxes(black_image, anchors, x=15, y=15, level=1, num_boxes=1, color=(255, 0, 0))
    add_boxes(black_image, anchors, x=15, y=16, level=1, num_boxes=1, color=(0, 255, 0))

    plt.imshow(black_image)
    plt.show()


def test_vgg():
    model = Backbone.ssd_vgg16(debug=False)
    # model = tiny_base_net()
    with torch.no_grad():
        input_size = 300
        image = torch.zeros((1, 3, input_size, input_size))
        embedding = model(image)

    # print(model)

    for feature_map in embedding:
        print('\nfeature map shape:', feature_map.shape)


def test_torchvision():
    model = ssd.ssd300_vgg16(pretrained_backbone=False)
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
    # test_vgg()
    # test_torchvision()
