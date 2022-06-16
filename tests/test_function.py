import torch
from skimage.io import imread

from utils.bounding_boxes import create_anchor_boxes
from utils.funcs import draw_boxes, show_image


SIMPLE_SCALES = True
torch.set_printoptions(2)


def main():
    data = torch.tensor(imread('res/catdog.png'))

    if SIMPLE_SCALES:
        scales = [0.3]
        ratios = [1, 2, 0.5]
    else:
        scales = [0.75, 0.5, 0.25]
        ratios = [1, 2, 0.5]

    boxes_own = create_anchor_boxes((3, 3), scales=scales, ratios=ratios, device='cuda')
    boxes_own = boxes_own.reshape((-1, 4))

    image = data.clone()
    draw_boxes(image, boxes_own, colors='random')
    show_image(image)


if __name__ == '__main__':
    main()
