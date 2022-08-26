from pathlib import Path

import numpy as np
import torch
from determined.pytorch import DataLoader

from datasets import LizardDetectionDataset
from datasets.banana_dataset import load_data_bananas
from utils.funcs import debug


def get_image(ds_name, batch):
    if ds_name == 'lizard':
        return batch['image']
    if ds_name == 'banana':
        return batch[0]


def get_bboxes(ds_name, batch):
    if ds_name == 'lizard':
        return batch['boxes']
    if ds_name == 'banana':
        return batch[1]


def get_lizard(image_size):
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([image_size, image_size]),
        image_stride=np.array([image_size, image_size]),
        use_cache=True,
        show_progress=True,
    )

    # train_dataset, eval_dataset = dataset.split(0.5)
    train_dataset = dataset

    print(f'len train: {len(train_dataset)}')
    train_data_loader = DataLoader(train_dataset, batch_size=32)

    return train_data_loader, 6


def get_banana():
    train_data_loader, _ = load_data_bananas('../data/banana-detection', 32, verbose=False)
    return train_data_loader, 1


def main():
    for ds_name in ('lizard', 'banana'):
        print('----- {} -----'.format(ds_name))

        if ds_name == 'lizard':
            dataloader, num_classes = get_lizard(256)
        elif ds_name == 'banana':
            dataloader, num_classes = get_banana()
        else:
            raise ValueError('Unknown dataset: {}'.format(ds_name))

        for batch in dataloader:
            image = get_image(ds_name, batch)
            boxes = get_bboxes(ds_name, batch)
            debug(image.shape)
            debug(boxes.shape)
            debug(torch.max(image[0]))
            debug(torch.min(image[0]))

            class_labels = boxes[:, :, 0].type(torch.int)
            debug(class_labels.shape)
            # debug(class_labels)
            debug(torch.bincount(class_labels.flatten() + 1))
            break


if __name__ == '__main__':
    main()
