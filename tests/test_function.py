from d2l import torch as d2l
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
    image_src = image_src.permute(2, 0, 1)

    scales = [0.1, 0.2]
    ratios = [1.0, 0.5, 2.0]
    # scales = [0.1]
    # ratios = [1.0]

    anchor_boxes_own = create_anchor_boxes(
        image_src.shape[-2:], scales=scales, ratios=ratios, device='cpu'
    )
    anchor_boxes_lib = d2l.multibox_prior(image_src, sizes=scales, ratios=ratios)

    debug(image_src.shape)
    print('anchor_boxes_own:\n', anchor_boxes_own)
    print('anchor_boxes_lib:\n', anchor_boxes_lib)
    print('same:', (anchor_boxes_own == anchor_boxes_lib).all())
    print('similar:', (anchor_boxes_own - anchor_boxes_lib).abs().max())
    print('lib.shape:', anchor_boxes_lib.shape, 'own.shape:', anchor_boxes_own.shape)

    # cmp(anchor_boxes_lib, anchor_boxes_local)
    # cmp(anchor_boxes_lib, anchor_boxes_own)
    # debug(anchor_boxes.shape)
    # debug(anchor_boxes)
    # debug(anchor_boxes_own[0])


if __name__ == '__main__':
    main()
