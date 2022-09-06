import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.ops.boxes import _box_inter_union

from models import SSDModel
from datasets.lizard_detection import LizardDetectionDataset
from utils.bounding_boxes import multibox_target, generate_random_boxes, intersection_over_union2, \
    intersection_over_union, box_iou
from utils.clock import Clock
from utils.funcs import debug
from torchvision.ops import box_iou as ops_iou
from tqdm import tqdm
from torchmetrics import JaccardIndex

BATCH_SIZE = 3
NUM_CLASSES = 6
torch.set_printoptions(2)
SMOOTH = 1e-6


def main():
    model = SSDModel(num_classes=NUM_CLASSES, backbone_arch='vgg16', min_anchor_size=0.2, max_anchor_size=0.9)
    dataset = LizardDetectionDataset.from_avocado(
        image_size=np.array([300, 300]),
        use_cache=True,
        show_progress=True
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    for batch in data_loader:
        images = batch['image']
        boxes = batch['boxes']
        # create data
        anchors, cls_preds, bbox_preds = model(images)
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, boxes)

        # reshape
        bbox_preds = bbox_preds.reshape(BATCH_SIZE, -1, 4)
        bbox_masks = bbox_masks.reshape(BATCH_SIZE, -1, 4)
        bbox_labels = bbox_labels.reshape(BATCH_SIZE, -1, 4)

        debug(images.shape)
        debug(boxes.shape)
        debug(anchors.shape)
        debug(cls_preds.shape)
        debug(bbox_preds.shape)
        debug(bbox_masks.shape)
        debug(cls_labels.shape)
        debug(bbox_labels.shape)

        # noinspection PyUnreachableCode
        if True:
            for mask, cls_label in zip(bbox_masks[0, :, 0], cls_labels[0]):
                if mask != 0.0 or cls_label != 0.0:
                    print('mask: {}  cls_label: {}'.format(mask, cls_label))
        break


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    def intersect(box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersection_over_union_grid(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    L = 0; T = 1; R = 2; B = 3

    # clock = Clock()

    # iboxes1[i] yields the smallest grid-cell-box wrapping it
    # iboxes1 = torch.clamp(boxes1, min=0, max=1.0-0.0001).int()
    # iboxes2 = torch.clamp(boxes2, min=0, max=1.0-0.0001).int()

    # clock.stop_and_print('init: {} seconds')

    iou = torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    grid1 = [[0, 0], [0, 0]]
    grid2 = [[0, 0], [0, 0]]
    # clock.stop_and_print('create grid: {} seconds')

    # grid1[x][y] contains the indices of the boxes1 intersecting that quadrant.
    # grid1[0][0] is left-top
    grid1[0][0] = torch.nonzero(torch.logical_and(boxes1[:, L] <= 0.5, boxes1[:, T] <= 0.5))
    grid1[0][1] = torch.nonzero(torch.logical_and(boxes1[:, L] <= 0.5, boxes1[:, B] >= 0.5))
    grid1[1][0] = torch.nonzero(torch.logical_and(boxes1[:, R] >= 0.5, boxes1[:, T] <= 0.5))
    grid1[1][1] = torch.nonzero(torch.logical_and(boxes1[:, R] >= 0.5, boxes1[:, B] >= 0.5))

    grid2[0][0] = torch.nonzero(torch.logical_and(boxes2[:, L] <= 0.5, boxes2[:, T] <= 0.5))
    grid2[0][1] = torch.nonzero(torch.logical_and(boxes2[:, L] <= 0.5, boxes2[:, B] >= 0.5))
    grid2[1][0] = torch.nonzero(torch.logical_and(boxes2[:, R] >= 0.5, boxes2[:, T] <= 0.5))
    grid2[1][1] = torch.nonzero(torch.logical_and(boxes2[:, R] >= 0.5, boxes2[:, B] >= 0.5))

    # clock.stop_and_print('fill grid: {} seconds')

    for x in range(2):
        for y in range(2):
            grid1[x][y] = torch.squeeze(grid1[x][y])
            grid2[x][y] = torch.squeeze(grid2[x][y])
            ij_pairs = torch.cartesian_prod(grid1[x][y], grid2[x][y])
            box1 = boxes1[ij_pairs[:, 0]]
            box2 = boxes2[ij_pairs[:, 1]]

            left = torch.maximum(box1[:, L], box2[:, L])
            right = torch.minimum(box1[:, R], box2[:, R])
            top = torch.maximum(box1[:, T], box2[:, T])
            bot = torch.minimum(box1[:, B], box2[:, B])
            z = torch.tensor(0.0)
            intersection_area = torch.maximum(z, right - left) * torch.maximum(z, bot-top)
            box1_area = (box1[:, R] - box1[:, L]) * (box1[:, B] - box1[:, T])
            box2_area = (box2[:, R] - box2[:, L]) * (box2[:, B] - box2[:, T])
            results = intersection_area / (box1_area + box2_area - intersection_area)

            iou[ij_pairs[:, 0], ij_pairs[:, 1]] = results
    # clock.stop_and_print('iou {} seconds')

    return iou


def torch_part(boxes1, boxes2):
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def test_iou():
    num_boxes = 2000
    boxes1 = generate_random_boxes(num_boxes, min_size=0.02, max_size=0.04)
    boxes2 = generate_random_boxes(num_boxes, min_size=0.02, max_size=0.04)
    # debug(torch.mean(boxes1-boxes2))
    clock = Clock()
    # iou_new = intersection_over_union2(boxes1, boxes2)
    # clock.stop_and_print('new: {} seconds')
    fs = [intersection_over_union, box_iou, ops_iou, intersection_over_union_grid, torch_part]
    results = []
    iou_compare = ops_iou(boxes1, boxes2)
    for f in tqdm(fs):
        print('check', f.__name__)
        clock = Clock()
        iou = None
        for _ in range(10):
            iou = f(boxes1, boxes2)
        duration = clock.stop()
        assert torch.allclose(iou, iou_compare)
        results.append((f.__name__, duration))

    results = sorted(results, key=lambda x: x[1])
    for result in results:
        print('{}: {}'.format(*result))


if __name__ == '__main__':
    test_iou()
    # main()
