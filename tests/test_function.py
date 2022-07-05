import sys

import torch

from utils.bounding_boxes import generate_random_boxes, intersection_over_union, tlbr_to_yxhw, yxhw_to_tlbr, \
    assign_anchor_to_ground_truth_boxes, create_anchor_boxes, multibox_target
# noinspection PyUnresolvedReferences
from skimage.io import imread

from utils.clock import Clock
# noinspection PyUnresolvedReferences
from utils.funcs import draw_boxes, show_image, debug

NUM_SAMPLES1 = 2
torch.set_printoptions(2)


def main():
    image_src = torch.tensor(imread('res/black512.png'))

    anchor_boxes = create_anchor_boxes((10, 10), scales=[0.1, 0.2], ratios=[1.0, 0.5, 2.0])
    anchor_boxes = anchor_boxes.reshape((-1, 4))
    ground_truth = torch.tensor([
        [0, 0.1, 0.1, 0.3, 0.3],
        [1, 0.4, 0.4, 0.5, 0.5],
    ])
    ground_truth[:, 1:] += ((torch.randn((2, 4)) - 0.5) * 0.02)
    debug(ground_truth)

    draw_boxes(image_src, bounding_boxes=anchor_boxes, color=128)
    draw_boxes(image_src, bounding_boxes=ground_truth[:, 1:], color=255)
    show_image(image_src)

    _, _, labels = multibox_target(anchor_boxes.unsqueeze(0), ground_truth.unsqueeze(0))
    debug(labels)


def profile():
    import cProfile
    import pstats
    import io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

    boxes1 = generate_random_boxes(NUM_SAMPLES1)
    boxes2 = generate_random_boxes(NUM_SAMPLES1+1000)
    # ------------------------
    intersection_over_union(boxes1, boxes2)
    # ------------------------
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()
    print(s.getvalue())

    sys.exit(0)


if __name__ == '__main__':
    main()
