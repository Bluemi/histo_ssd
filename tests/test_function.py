import torch
import numpy as np
from torch.utils.data import DataLoader

from models import SSDModel
from datasets.lizard_detection import LizardDetectionDataset
from utils.bounding_boxes import multibox_target, generate_random_boxes, intersection_over_union2, \
    intersection_over_union
from utils.clock import Clock
from utils.funcs import debug

BATCH_SIZE = 3
NUM_CLASSES = 6
torch.set_printoptions(2)


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


def test_iou():
    boxes1 = generate_random_boxes(8000, min_size=0.02, max_size=0.04)
    boxes2 = generate_random_boxes(8000, min_size=0.02, max_size=0.04)
    # debug(torch.mean(boxes1-boxes2))
    clock = Clock()
    iou_new = intersection_over_union2(boxes1, boxes2)
    clock.stop_and_print('new: {} seconds')
    iou_orig = intersection_over_union(boxes1, boxes2)
    clock.stop_and_print('orig: {} seconds')

    debug(torch.max(iou_new - iou_orig))

    # debug(intersection_over_union(boxes1, boxes2))
    # debug()
    return
    print('calc iou', flush=True)
    iou1 = intersection_over_union(boxes1, boxes2)
    print('Done', flush=True)
    print('calc iou2', flush=True)
    iou2 = intersection_over_union2(boxes1, boxes2)
    print('Done', flush=True)
    print(torch.mean(iou1 - iou2))


if __name__ == '__main__':
    test_iou()
    # main()
