import sys

import torch

from utils.bounding_boxes import generate_random_boxes, intersection_over_union, tlbr_to_yxhw, yxhw_to_tlbr
# noinspection PyUnresolvedReferences
from skimage.io import imread

from utils.clock import Clock
# noinspection PyUnresolvedReferences
from utils.funcs import draw_boxes, show_image

NUM_SAMPLES1 = 2
torch.set_printoptions(2)


def main():
    # profile()

    boxes = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 4]], dtype=torch.float)

    yxhw = tlbr_to_yxhw(boxes)
    tlbr = yxhw_to_tlbr(yxhw)

    print(boxes)
    print(yxhw)
    print(tlbr)


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
