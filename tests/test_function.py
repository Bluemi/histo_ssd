import sys

import torch

from utils.bounding_boxes import generate_random_boxes, intersection_over_union, tlbr_to_yxhw, yxhw_to_tlbr, \
    assign_anchor_to_ground_truth_boxes, create_anchor_boxes
# noinspection PyUnresolvedReferences
from skimage.io import imread

# noinspection PyUnresolvedReferences
from utils.funcs import draw_boxes, show_image, debug

NUM_SAMPLES1 = 2
torch.set_printoptions(2)


def main():
    image_src = torch.tensor(imread('res/black.png'))

    debug(image_src.shape)
    anchor_boxes = create_anchor_boxes(image_src.shape[-2:], scales=[0.1, 0.2], ratios=[1.0, 0.5, 2.0], device=image_src.device)
    debug(anchor_boxes.shape)


if __name__ == '__main__':
    main()
