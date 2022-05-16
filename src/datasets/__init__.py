import numpy as np
from torch.utils.data import Dataset
from lizard_detection import LizardDetectionDataset


def get_dataset(dataset_name: str) -> Dataset:
    dataset = None
    if dataset_name == 'lizard':
        dataset = LizardDetectionDataset.from_avocado(
            image_size=np.ndarray([224, 224]),
            image_stride=np.ndarray([112, 112]),
        )
    if dataset is None:
        raise ValueError('unknown dataset \"{}\"'.format(dataset_name))
    return dataset
