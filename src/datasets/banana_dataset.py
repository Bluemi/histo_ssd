import os
import pandas as pd
import torch
import torch.utils.data
import torchvision


def read_data_bananas(data_dir: str, is_train=True, verbose=True):
    """Read the banana detection dataset images and labels."""
    if verbose:
        print(data_dir)
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        image = torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')
        )
        image = image.float() / 255.0  # normalize images
        images.append(image)
        # Here `target` contains (class, upper-left x, upper-left y, lower-right x, lower-right y),
        # where all the images have the same banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, data_dir: str, is_train, verbose=True, transforms=None):
        self.features, self.labels = read_data_bananas(data_dir, is_train, verbose=verbose)
        self.transforms = transforms or []
        if verbose:
            print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    @staticmethod
    def normalization_values():
        # Dont normalize banana dataset
        mean = [0.0, 0.0, 0.0]
        std = [1.0,  1.0, 1.0]
        return mean, std

    def __getitem__(self, idx):
        image = self.features[idx].float()
        for transform in self.transforms:
            image = transform(image)
        return {
            'image': image,
            'boxes': self.labels[idx],
        }

    def __len__(self):
        return len(self.features)


def load_data_bananas(data_dir: str, batch_size, verbose=True):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(
        BananasDataset(data_dir, is_train=True, verbose=verbose),
        batch_size,
        shuffle=True
    )
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(data_dir, is_train=False, verbose=verbose),
        batch_size
    )
    return train_iter, val_iter
