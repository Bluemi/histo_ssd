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


SHOW_IMAGE = True


def main():
    ignore_classes = [0, 4]
    # ignore_classes = None
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([224, 224]),
        image_stride=np.array([224, 224]),
        use_cache=True,
        show_progress=True,
        ignore_classes=ignore_classes,
    )

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

    data_loader = DataLoader(dataset, batch_size=64)

    label_distribution = defaultdict(int)
    sample_counter = 0
    for batch in data_loader:
        for image, sample in zip(batch['image'], batch['boxes']):
            sample_counter += 1
            for box in sample:
                label = box[0].item()
                label_distribution[label] += 1
            if SHOW_IMAGE:
                image = (image.permute((1, 2, 0)) * 255.0).to(torch.int32)
                draw_boxes(image, sample[:, 1:], box_format='ltrb')
                plt.imshow(image)
                plt.show()

    for label in sorted(label_distribution.keys()):
        print('{}: {}'.format(label, label_distribution[label]))
    print('{} samples'.format(sample_counter))


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
    # profile()
