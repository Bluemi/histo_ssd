import sys

import torch
import cProfile

from utils.bounding_boxes import create_random_boxes, intersection_over_union
from skimage.io import imread

from utils.clock import Clock
from utils.funcs import draw_boxes, show_image

NUM_SAMPLES1 = 5200
NUM_SAMPLES2 = 5500
torch.set_printoptions(2)


def main():
    # image = torch.tensor(imread('res/black1024x1024.png'))

    # profile()

    boxes1 = create_random_boxes(NUM_SAMPLES1, device='cuda')
    boxes2 = create_random_boxes(NUM_SAMPLES2, device='cuda')

    clock = Clock()
    clock.start()
    result_own = intersection_over_union(boxes1, boxes2)
    clock.stop_and_print('own time: {} sec')

    print('\nresult_own')
    print(result_own)

    # show_image(image)


def profile():
    import cProfile
    import pstats
    import io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

    boxes1 = create_random_boxes(NUM_SAMPLES1)
    boxes2 = create_random_boxes(NUM_SAMPLES2)
    # ------------------------
    intersection_over_union(boxes1, boxes2)
    # ------------------------
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    sys.exit(0)


if __name__ == '__main__':
    main()
