from collections import defaultdict
from pathlib import Path
import cProfile
from pstats import SortKey

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from datasets.augmentation_wrapper import AugmentationWrapper
from datasets.lizard_detection import LizardDetectionDataset
from utils.augmentations import RandomRotate, RandomFlip
from utils.bounding_boxes import box_area
from utils.funcs import draw_boxes, DEFAULT_COLORS1, DARK_COLORS, BRIGHT_COLORS
from utils.metrics import box_indices_inside

SHOW_IMAGE = False


def main():
    # ignore_classes = [0, 4]
    ignore_classes = None
    whole_dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([300, 300]),
        image_stride=np.array([150, 150]),
        use_cache=True,
        show_progress=True,
        ignore_classes=ignore_classes,
        ignore_image=False,
    )

    # train, validation = whole_dataset.split(0.8)

    # debug(len(whole_dataset))
    # debug(len(train))
    # debug(len(validation))

    # show_max_boxes(whole_dataset)
    show_area_stats(whole_dataset)
    # show_images(whole_dataset)
    # show_distributions(whole_dataset, train, validation)
    # test_cut(whole_dataset)
    # test_box_plotting(whole_dataset)


def test_box_plotting(dataset):
    for i in trange(len(dataset)):
        sample = dataset[i]
        image = sample['image']
        boxes = filter_boxes(sample['boxes'])
        unique_values = np.unique(boxes[:, 0])
        if len(unique_values) == 6:
            image = (image.permute((1, 2, 0)) * 255.0).to(torch.int32)
            # boxes_third = len(boxes) // 3
            boxes1 = boxes
            draw_boxes(
                image, torch.tensor(boxes1[:, 1:]), box_format='ltrb',
                color=DARK_COLORS, color_indices=torch.tensor(boxes1[:, 0]), sign='box', color_mode='set',
            )
            boxes2 = np.copy(boxes) # np.copy(boxes[:boxes_third*2])
            boxes2[:, 1:] += (np.random.rand(boxes2.shape[0], 4)-0.5)*0.02
            draw_boxes(
                image, torch.tensor(boxes2[:, 1:]), box_format='ltrb',
                color=BRIGHT_COLORS, color_indices=torch.tensor(boxes2[:, 0]), sign='box', color_mode='set'
            )
            plt.imshow(image)
            plt.show()


def wrap_dataset(dataset):
    return AugmentationWrapper(
        dataset,
        [
            (None, RandomRotate()),
            (None, RandomFlip())
        ]
    )


def test_cut(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image']
        boxes = filter_boxes(sample['boxes'])
        boxes = torch.tensor(boxes)
        box_indices = box_indices_inside(boxes[:, 1:], torch.tensor([0.0, 0.0, 0.5, 0.5]))
        boxes = boxes[box_indices]
        show_sample(image, boxes)


def show_sample(image, boxes):
    image = (image.permute((1, 2, 0)) * 255.0).to(torch.int32)
    boxes = filter_boxes(boxes)
    draw_boxes(
        image, torch.tensor(boxes[:, 1:]), box_format='ltrb',
        color=BRIGHT_COLORS, color_indices=torch.tensor(boxes[:, 0]), sign='box', color_mode='set',
    )
    plt.imshow(image)
    plt.show()


def filter_boxes(boxes):
    valid_box_indices = boxes[:, 0] != -1.0  # filter out invalid boxes
    return boxes[valid_box_indices]


def show_images(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image']
        boxes = filter_boxes(sample['boxes'])
        show_sample(image, boxes)


def show_area_stats(dataset):
    max_area = 0
    min_area = 1

    box_area_sum = 0
    num_boxes = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        boxes = filter_boxes(sample['boxes'])

        box_areas = box_area(boxes[:, 1:])

        box_area_sum += np.sum(box_areas)
        num_boxes += len(box_areas)
        max_area = max(max_area, np.max(box_areas))
        min_area = min(min_area, np.min(box_areas))

    print('min area:', min_area)
    print('max area:', max_area)
    print('mean area:', box_area_sum / num_boxes)
    print('avg num objects per image:', num_boxes / len(dataset))


def show_distributions(whole_dataset, train_dataset, val_dataset):
    label_distribution = get_distribution(whole_dataset)
    print('whole dataset')
    print_distribution(label_distribution)

    train_label_distribution = get_distribution(train_dataset)
    print('train dataset')
    print_distribution(train_label_distribution)

    val_label_distribution = get_distribution(val_dataset)
    print('val dataset')
    print_distribution(val_label_distribution)


def print_distribution(label_distribution: dict):
    for label in sorted(label_distribution.keys()):
        print('{}: {}'.format(label, label_distribution[label]))
    print('total:', sum(label_distribution.values()))


def get_distribution(dataset):
    label_distribution = defaultdict(int)
    sample_counter = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        boxes = filter_boxes(sample['boxes'])
        sample_counter += 1
        for box in boxes:
            label = box[0].item()
            label_distribution[label] += 1
    if -1 in label_distribution:
        del label_distribution[-1]
    return label_distribution


def show_max_boxes(dataset):
    max_length = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        boxes = filter_boxes(sample['boxes'])
        max_length = max(max_length, len(boxes))
        if len(boxes) >= 600:
            print('num boxes:', len(boxes))
            show_sample(sample['image'], boxes)
    print('max boxes:', max_length)


def profile():
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([224, 224]),
        image_stride=np.array([224, 224]),
        use_cache=True,
        show_progress=True,
    )
    pr = cProfile.Profile()
    pr.enable()
    pr.disable()
    pr.print_stats(sort=SortKey.CUMULATIVE)


if __name__ == '__main__':
    main()
    # profile()
