"""
Lizard Dataset described here: https://arxiv.org/pdf/2108.11195.pdf

TODO: Cache label data
"""

from pathlib import Path
from typing import List, Dict, Any

import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


AVOCADO_DATASET_LOCATION = Path('/data/ldap/histopathologic/original_read_only/Lizard')
LABELS_DIR = Path('labels/Labels')


def imread(image_path):
    with Image.open(image_path) as image:
        # noinspection PyTypeChecker
        return np.array(image)


class Snapshot:
    """
    A Snapshot stands for a subregion in an image. It only saves positional information and no data itself
    (no image or label data):
    - The sample name: if the image file is named "images1/consep_10.png" the sample name is consep_10.
    - The image directory: either images1 or images2
    - The position of the subimage inside the image
    """

    def __init__(self, sample_name: str, image_directory: str, position: np.ndarray):
        self.sample_name = sample_name
        self.image_directory = Path(image_directory)
        self.position = position

    def get_image_path(self) -> Path:
        return self.image_directory / f'{self.sample_name}.png'

    def get_label_path(self) -> Path:
        return LABELS_DIR / f'{self.sample_name}.mat'

    def __repr__(self):
        return f'Snapshot(sample_name={self.sample_name} position=(y={self.position[0]}, x={self.position[1]})'


class LizardDetectionDataset(Dataset):
    def __init__(self, snapshots: List[Snapshot], data_dir: Path, image_size: np.ndarray, cache_images: bool = False):
        """
        Args:
            snapshots: List of snapshots for this dataset
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            cache_images: Whether to keep loaded images in memory. Defaults to False.
        """
        self.snapshots = snapshots
        self.data_dir = data_dir
        self.image_size = image_size
        self.cache_images = cache_images
        self.image_cache: Dict[Path, np.ndarray] = {}

    @staticmethod
    def from_datadir(data_dir: Path, image_size: np.ndarray, image_stride: np.ndarray or None = None):
        """
        Args:
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
        """
        image_stride = image_stride or image_size  # use image size as image stride, if no images stride provided
        snapshots = LizardDetectionDataset._define_snapshots(data_dir, image_size, image_stride)
        return LizardDetectionDataset(
            snapshots=snapshots,
            data_dir=data_dir,
            image_size=image_size,
        )

    @staticmethod
    def from_avocado(image_size: np.ndarray, image_stride: np.ndarray or None = None):
        """
        Args:
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
        """
        return LizardDetectionDataset.from_datadir(
            data_dir=AVOCADO_DATASET_LOCATION,
            image_size=image_size,
            image_stride=image_stride,
        )

    @staticmethod
    def _define_snapshots(data_dir: Path, image_size: np.ndarray, image_stride: np.ndarray) -> List[Snapshot]:
        """
        Returns a list of snapshots to use for this dataset. Images will be searched in '<data_dir>/images1/*.png' and
        '<data_dir>/images2/*.png'.
        """
        image1_files = sorted([f for f in (data_dir / 'images1').iterdir() if str(f).endswith('.png')])
        image2_files = sorted([f for f in (data_dir / 'images2').iterdir() if str(f).endswith('.png')])
        image_files = image1_files + image2_files

        subimages = []
        for image_file in image_files:
            subimages.extend(LizardDetectionDataset.snapshots_from_image_file(image_file, image_size, image_stride))

        return subimages

    @staticmethod
    def snapshots_from_image_file(filename: Path, image_size: np.ndarray, image_stride: np.ndarray) -> List[Snapshot]:
        with Image.open(filename) as image:
            width, height = image.size
        full_image_size = np.array([height, width])
        sample_name = filename.stem
        image_dir = filename.parent.stem
        position = np.array([0, 0])  # The position of the current snapshot as (y, x)
        snapshots = []
        # iterate as long as right-bottom corner of subimage is in bounds of full_image_size
        while (position + image_size <= full_image_size).all():
            snapshot = Snapshot(sample_name, image_dir, position.copy())
            snapshots.append(snapshot)

            # move subimage to the right
            position[1] += image_stride[1]
            if position[1] + image_size[1] > full_image_size[1]:
                # move to next line
                position[1] = 0
                position[0] += image_stride[0]
        return snapshots

    def read_image(self, image_path):
        if self.cache_images:
            image = self.image_cache.get(image_path)
            if image is None:
                image = imread(str(image_path))
                self.image_cache[image_path] = image
        else:
            image = imread(str(image_path))
        return image

    def get_subimage(self, snapshot) -> np.ndarray:
        image_path = self.data_dir / snapshot.get_image_path()
        image = self.read_image(image_path)
        return image[
            snapshot.position[0]:snapshot.position[0]+self.image_size[0],
            snapshot.position[1]:snapshot.position[1]+self.image_size[1]
        ]

    def get_label_data(self, snapshot) -> Dict:
        label_path = self.data_dir / snapshot.get_label_path()
        return sio.loadmat(str(label_path))

    @staticmethod
    def transform_bounding_box(bounding_box, snapshot) -> np.ndarray or None:
        """
        Args:
            bounding_box: The bounding box to transform as (y1, y2, x1, x2)
            snapshot: The snapshot whose position is used
        Returns:
            The bounding box in form (y1, x1, y2, x2) where x and y coordinates are subtracted with the snapshot
            position.
        """
        return np.array([
            bounding_box[0] - snapshot.position[0],  # y1
            bounding_box[2] - snapshot.position[1],  # x1
            bounding_box[1] - snapshot.position[0],  # y2
            bounding_box[3] - snapshot.position[1],  # x2
        ])

    def _bounding_box_in_snapshot(self, bounding_box: np.ndarray) -> bool:
        # check lower bound
        if (bounding_box < 0).any():
            return False
        # check y upper bound
        if bounding_box[0] > self.image_size[0] or bounding_box[2] > self.image_size[0]:
            return False
        # check x upper bound
        if bounding_box[1] > self.image_size[1] or bounding_box[3] > self.image_size[1]:
            return False
        return True

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Gets the dataset sample with the given index. Every sample contains N instances of nuclei.
        A sample is a dictionary with the following keys:
          - image: An image with shape [height, width, 3]
          - labels: A List[N] of class numbers 0 <= label <= 5
          - boxes: A List[N, 4] containing the bounding boxes of the sample as [top, left, bottom, right]
        """
        snapshot = self.snapshots[index]

        subimage = self.get_subimage(snapshot)
        label_data = self.get_label_data(snapshot)

        inst_map = label_data['inst_map']
        # noinspection PyTypeChecker
        nuclei_id: list = np.squeeze(label_data['id']).tolist()
        unique_values = np.unique(inst_map).tolist()[1:]  # remove 0 from begin; I checked and 0 is always at begin
        bounding_boxes = []
        instance_classes = []
        for value in unique_values:
            idx = nuclei_id.index(value)
            # bounding box
            bounding_box = label_data['bbox'][idx]
            bounding_box = LizardDetectionDataset.transform_bounding_box(bounding_box, snapshot)
            if self._bounding_box_in_snapshot(bounding_box):
                # check whether bounding box is in snapshot
                bounding_boxes.append(bounding_box)
                # class
                instance_class = label_data['class'][idx][0]  # checked. Is always a list with exactly one element.
                instance_classes.append(instance_class)
        bounding_boxes = np.stack(bounding_boxes) if bounding_boxes else np.array([])
        instance_classes = np.stack(instance_classes) if instance_classes else np.array([])
        return {
            'image': subimage,
            'boxes': bounding_boxes,
            'labels': instance_classes,
        }

    def __len__(self):
        return len(self.snapshots)

    @staticmethod
    def collate_fn(batch):
        images = []
        boxes = []
        labels = []
        for sample in batch:
            images.append(sample['image'])
            boxes.append(sample['boxes'])
            labels.append(sample['labels'])

        images = torch.Tensor(np.stack(images))
        return {
            'image': images,
            'boxes': boxes,
            'labels': labels,
        }
