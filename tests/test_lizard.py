from collections import defaultdict
from pathlib import Path
import cProfile
from pstats import SortKey

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from datasets.augmentation_wrapper import AugmentationWrapper
from datasets.lizard_detection import LizardDetectionDataset
from utils.augmentations import RandomRotate, RandomFlip
from utils.clock import Clock
from utils.funcs import debug, draw_boxes


SHOW_IMAGE = False


def main():
    # ignore_classes = [0, 4]
    ignore_classes = None
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([300, 300]),
        image_stride=np.array([300, 300]),
        use_cache=True,
        show_progress=True,
        ignore_classes=ignore_classes,
    )

    debug(len(dataset))

    train, validation = dataset.split(0.8)

    debug(len(train))
    debug(len(validation))

    dataset = AugmentationWrapper(
        dataset,
        [
            (None, RandomRotate()),
            (None, RandomFlip())
        ]
    )

    label_distribution = get_distribution(dataset)
    print('whole dataset')
    print_distribution(label_distribution)

    train_label_distribution = get_distribution(train)
    print('train dataset')
    print_distribution(train_label_distribution)

    val_label_distribution = get_distribution(validation)
    print('val dataset')
    print_distribution(val_label_distribution)

    max_boxes = 0
    min_boxes = 100000
    for i in range(len(dataset)):
        sample = dataset[i]
        boxes = sample['boxes']
        print('num boxes:', len(boxes))
        max_boxes = max(max_boxes, len(boxes))
        min_boxes = min(min_boxes, len(boxes))

    print('max boxes:', max_boxes)
    print('min boxes:', min_boxes)

    if SHOW_IMAGE:
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample['image']
            boxes = sample['boxes']
            image = (image.permute((1, 2, 0)) * 255.0).to(torch.int32)
            draw_boxes(image, torch.tensor(boxes[:, 1:]), box_format='ltrb')
            plt.imshow(image)
            plt.show()


def main2():
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([300, 300]),
        image_stride=np.array([300, 300]),
        use_cache=True,
        show_progress=True,
        ignore_classes=[0, 4],
    )
    train_set, val_set = dataset.split(0.8)
    print('len val:', len(val_set))


def print_distribution(label_distribution: dict):
    for label in sorted(label_distribution.keys()):
        print('{}: {}'.format(label, label_distribution[label]))
    print('total:', sum(label_distribution.values()))


def get_distribution(dataset):
    label_distribution = defaultdict(int)
    sample_counter = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        boxes = sample['boxes']
        sample_counter += 1
        for box in boxes:
            label = box[0].item()
            label_distribution[label] += 1
    del label_distribution[-1]
    return label_distribution


def profile():
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([224, 224]),
        image_stride=np.array([224, 224]),
        use_cache=True,
        show_progress=True,
    )
    data_loader = DataLoader(dataset, batch_size=64)
    pr = cProfile.Profile()
    pr.enable()
    iterate_through(data_loader)
    pr.disable()
    pr.print_stats(sort=SortKey.CUMULATIVE)


def iterate_through(data_loader):
    clock = Clock()
    for _batch in data_loader:
        pass
    clock.stop_and_print('took {} seconds')


if __name__ == '__main__':
    main()
    # main2()
    # profile()
