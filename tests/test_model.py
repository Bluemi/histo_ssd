from collections import OrderedDict

import torch
from matplotlib import pyplot as plt
from torch.hub import load_state_dict_from_url

from models import SSDModel, Backbone
from utils.funcs import draw_boxes, debug
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
BATCH_SIZE = 7


KEY_MAPPING = OrderedDict([
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


def main():
    model = SSDModel(num_classes=1, debug=True, backbone_arch=MODEL)

    print(model)
    return

    model.eval()

    with torch.no_grad():
        image = torch.zeros((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))
        x = image
        for block in model.backbone.blocks:
            for layer in block:
                x = layer(x)
                print('{}: {}'.format(layer, x.shape))
        anchors, cls_preds, bbox_preds = model(image)

    anchors = anchors.squeeze(0)

    debug(anchors.shape)
    debug(cls_preds.shape)
    debug(bbox_preds.shape)
    debug(bbox_preds.reshape((BATCH_SIZE, -1, 4)).shape)

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
    state_dict = torch.load('../ssd300_vgg16_coco-b556d3b4.pth')
    # model_url = "https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth"
    # state_dict = load_state_dict_from_url(model_url, progress=True)
    remove_keys = []
    for key in state_dict.keys():
        if key.startswith('head.'):
            remove_keys.append(key)

    for remove_key in remove_keys:
        del state_dict[remove_key]

    new_state_dict = []
    for new_key, old_key in KEY_MAPPING.items():
        new_state_dict.append(
            (new_key, state_dict[old_key])
        )
    new_state_dict = OrderedDict(new_state_dict)

    # model = ssd.ssd300_vgg16(pretrained_backbone=False, num_classes=1)
    model = SSDModel(num_classes=6, backbone_arch='vgg16', debug=True)
    ret = model.load_state_dict(new_state_dict, strict=False)
    if ret.missing_keys:
        print('missing keys:')
    for missing_key in ret.missing_keys:
        print(missing_key)
    if ret.unexpected_keys:
        print('unexpected_key keys:')
    for unexpected_key in ret.unexpected_keys:
        print(unexpected_key)

    # my_model = SSDModel(num_classes=6, backbone_arch='vgg16', debug=True)
    # ret = my_model.load_state_dict(state_dict)
    # print(ret)

    # print(my_model)


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
    # main()
    # test_vgg()
    test_torchvision()
