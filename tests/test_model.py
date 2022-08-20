import torch
from matplotlib import pyplot as plt

from models import TinySSD, VGG
from utils.funcs import debug, draw_boxes
from torchvision.models.detection import ssd


def main():
    model = TinySSD(num_classes=1, debug=True)
    model.eval()

    base_model: torch.nn.Sequential = model.blocks[0]

    print(base_model._modules)

    with torch.no_grad():
        image = torch.zeros((32, 3, 256, 256))
        anchors, cls_preds, bbox_preds = model(image)

    anchors = anchors.squeeze(0)

    debug(anchors.shape)
    debug(cls_preds.shape)
    debug(bbox_preds.reshape((32, -1, 4)).shape)

    black_image = torch.zeros((1024, 1024, 3), dtype=torch.int)

    add_boxes(black_image, anchors, x=0.5, y=0.5, level=0, num_boxes=1, color=(255, 0, 0))
    # add_boxes(black_image, anchors, x=16, y=16, level=0,size=32, color=(0, 255, 0))

    plt.imshow(black_image)
    plt.show()


def test_vgg():
    model = VGG.ssd_vgg16(debug=True)
    with torch.no_grad():
        input_size = 300
        image = torch.zeros((1, 3, input_size, input_size))
        embedding = model(image)

    print('\nembedding shape:', embedding.shape)


def test_torchvision():
    model = ssd.ssd300_vgg16(pretrained_backbone=False)
    print(model)


def add_boxes(image, anchors, x, y, level, color=(255, 0, 0), num_boxes=4):
    level_sizes = [32, 16, 8, 4, 1]
    level_add = 0
    for i in range(level):
        level_add += level_sizes[i] * level_sizes[i]

    if isinstance(x, float):
        x = int(x * level_sizes[level])
    if isinstance(y, float):
        y = int(y * level_sizes[level])

    index = level_add + y * level_sizes[level] + x

    draw_anchors = anchors[index*4:index*4+num_boxes]
    debug(draw_anchors.shape)

    draw_boxes(image, draw_anchors, box_format='ltrb', color=color)


if __name__ == '__main__':
    # main()
    test_vgg()
    # test_torchvision()
