import torch
import numpy as np
from torch.utils.data import DataLoader

from models import SSDModel
from datasets.lizard_detection import LizardDetectionDataset
from utils.bounding_boxes import multibox_target, generate_random_boxes, non_maximum_suppression
from utils.funcs import debug
from utils.metrics import ConfusionMatrix, update_confusion_matrix

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


def test_nms():
    N = 800
    boxes = generate_random_boxes(N)
    scores = torch.rand(N)
    for iou_threshold in [0.3, 0.5, 0.7]:
        keep_indices = non_maximum_suppression(boxes, scores, iou_threshold=iou_threshold)
        print('iou_t = {}: {} boxes'.format(iou_threshold, keep_indices.shape[0]))


def test_tmp():
    outer_box = torch.tensor([0.0, 0.0, 1.0, 1.0])
    outer_box_center = torch.tensor([
        (outer_box[0] + outer_box[2]) * 0.5,
        (outer_box[1] + outer_box[3]) * 0.5
    ])
    for x in range(2):
        for y in range(2):
            new_outer_box = torch.clone(outer_box)
            new_outer_box[[x*2, y*2+1]] = outer_box_center
            print(new_outer_box)


def test_confusion_matrix():
    ground_truth_boxes = torch.tensor([
        [0.0, 0.1, 0.1, 0.2, 0.2],
        [1.0, 0.22, 0.22, 0.32, 0.32],
        [2.0, 0.4, 0.4, 0.5, 0.5],  # false negative, same position, but different label
        [1.0, 0.9, 0.9, 1.0, 1.0],  # false negative, not matched by any
    ])

    predictions = torch.tensor([
        # label, conf, left, top, right, bottom
        [0.0, 1.0, 0.1, 0.1, 0.2, 0.2],  # true positive
        [1.0, 1.0, 0.2, 0.2, 0.3, 0.3],  # true positive
        [1.0, 1.0, 0.3, 0.3, 0.4, 0.4],  # false positive
        [1.0, 1.0, 0.4, 0.4, 0.5, 0.5],  # false positive, same position, but different label
        [0.0, 1.0, 0.1, 0.1, 0.2, 0.21],  # false positive, because gt box already covered by first pred
    ])
    confusion_matrix = ConfusionMatrix()
    update_confusion_matrix(confusion_matrix, ground_truth_boxes.unsqueeze(0), predictions.unsqueeze(0))
    print(confusion_matrix)


if __name__ == '__main__':
    # main()
    # test_nms()
    # test_tmp()
    test_confusion_matrix()
