from collections import defaultdict
from pathlib import Path
import cProfile

import numpy as np
from torch.utils.data import DataLoader

from datasets.lizard_detection import LizardDetectionDataset
from utils.funcs import debug


def main():
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([224, 224]),
        image_stride=np.array([224, 224]),
        use_cache=True,
        show_progress=True,
    )

    train, validation = dataset.split(0.8)

    debug(len(train))
    debug(len(validation))

    data_loader = DataLoader(dataset, batch_size=64)

    label_distribution = defaultdict(int)
    sample_counter = 0
    for batch in data_loader:
        for sample in batch['boxes']:
            sample_counter += 1
            for box in sample:
                label = box[0].item()
                label_distribution[label] += 1
    for label in sorted(label_distribution.keys()):
        print('{}: {}'.format(label, label_distribution[label]))
    print('{} samples'.format(sample_counter))


def profile():
    cProfile.run('LizardDetectionDataset.from_datadir(data_dir=Path("/home/alok/cbmi/data/LizardDataset"), image_size=np.array([224, 224]), image_stride=np.array([224, 224]), use_cache=True)')


if __name__ == '__main__':
    main()
