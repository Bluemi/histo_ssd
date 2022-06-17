import sys

import torch

from utils.bounding_boxes import generate_random_boxes, intersection_over_union, tlbr_to_yxhw, yxhw_to_tlbr, \
    assign_anchor_to_ground_truth_boxes
# noinspection PyUnresolvedReferences
from skimage.io import imread

from utils.clock import Clock
# noinspection PyUnresolvedReferences
from utils.funcs import draw_boxes, show_image

NUM_SAMPLES1 = 2
torch.set_printoptions(2)


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = intersection_over_union(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def main():
    # profile()

    anchor_boxes = generate_random_boxes(7)
    ground_truth_boxes = generate_random_boxes(5)

    assign_anchor_to_ground_truth_boxes(anchor_boxes, ground_truth_boxes)

    # result = assign_anchor_to_bbox(ground_truth_boxes, anchor_boxes, device='cpu')
    # print(result)


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
