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

    data_loader = DataLoader(validation, batch_size=32)

    for batch in data_loader:
        debug(batch['boxes'].shape)
        debug(batch['image'].shape)


def profile():
    cProfile.run('LizardDetectionDataset.from_datadir(data_dir=Path("/home/alok/cbmi/data/LizardDataset"), image_size=np.array([224, 224]), image_stride=np.array([224, 224]), use_cache=True)')


if __name__ == '__main__':
    main()
