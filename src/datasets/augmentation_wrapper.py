from typing import Any, List, Tuple, Optional

from torch.utils.data import Dataset


class AugmentationWrapper(Dataset):
    def __init__(self, dataset, transforms: List[Tuple[Optional[str], Any]]):
        """
        Creates a AugmentationWrapper dataset.
        :param dataset: The dataset to augment
        :param transforms: A list of transformations to apply on the dataset. Each entry is a tuple with values
                           (key, transform). If key is None, the transformation will be applied on the result of
                           dataset.__getitem__(). If key is not None the transformation will only be applied on the
                           given key.
        """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        x = self.dataset[index]
        for key, transform in self.transforms:
            if key is None:
                x = transform(x)
            else:
                x[key] = transform(x[key])
        return x

    def __len__(self):
        return len(self.dataset)
