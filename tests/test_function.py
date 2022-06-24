import sys

import torch

from utils.bounding_boxes import generate_random_boxes, intersection_over_union, tlbr_to_yxhw, yxhw_to_tlbr, \
    assign_anchor_to_ground_truth_boxes, create_anchor_boxes
# noinspection PyUnresolvedReferences
from skimage.io import imread

from utils.clock import Clock
# noinspection PyUnresolvedReferences
from utils.funcs import draw_boxes, show_image

NUM_SAMPLES1 = 2
torch.set_printoptions(2)


def main():
    image_src = torch.tensor(imread('res/black512.png'))
    # profile()
    anchor_boxes = create_anchor_boxes((4, 4), scales=[0.15, 0.19], ratios=[1.0, 0.5, 2.0])

    while True:
        ground_truth_boxes = generate_random_boxes(1, min_size=0.1, max_size=0.2)

        anchor_boxes = anchor_boxes.reshape((-1, 4))

        image = image_src.clone()
        draw_boxes(image, anchor_boxes)
        draw_boxes(image, ground_truth_boxes, color=(255, 255, 255))

        result = assign_anchor_to_ground_truth_boxes(anchor_boxes, ground_truth_boxes)
        print(result.shape)
        print(result)
        print(sorted(list(filter(lambda x: x != -1, result.numpy()))))

        show_image(image)


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
