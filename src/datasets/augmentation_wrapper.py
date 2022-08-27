from typing import Any, List

from torch.utils.data import Dataset


class AugmentationWrapper(Dataset):
    def __init__(self, dataset, transforms: List[Any]):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        x = self.dataset[index]
        for transform in self.transforms:
            x = transform(x)
        return x

    def __len__(self):
        return len(self.dataset)
