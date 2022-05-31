from pathlib import Path
import cProfile

import numpy as np

from datasets.lizard_detection import LizardDetectionDataset


def main():
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([224, 224]),
        image_stride=np.array([224, 224]),
        use_cache=True,
    )

    train, validation = dataset.split(0.8)

    print('train samples: {}  validation samples: {}'.format(len(train), len(validation)))
    train_sample_names = set(np.unique(list(map(lambda s: s.sample_name, train.snapshots))))
    validation_sample_names = set(np.unique(list(map(lambda s: s.sample_name, validation.snapshots))))
    print('train samples names:\n - {}'.format('\n - '.join(sorted(list(train_sample_names)))))
    print('validation samples names:\n - {}'.format('\n - '.join(sorted(list(validation_sample_names)))))
    print('has common: {}'.format(not train_sample_names.isdisjoint(sorted(list(validation_sample_names)))))


def profile():
    cProfile.run('LizardDetectionDataset.from_datadir(data_dir=Path("/home/alok/cbmi/data/LizardDataset"), image_size=np.array([224, 224]), image_stride=np.array([224, 224]), use_cache=True)')


if __name__ == '__main__':
    main()
