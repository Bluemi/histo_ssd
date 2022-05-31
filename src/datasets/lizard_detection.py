"""
Lizard Dataset described here: https://arxiv.org/pdf/2108.11195.pdf

TODO: split train/eval/test
TODO: rework transform bounding box
"""
import copy
import itertools
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


AVOCADO_DATASET_LOCATION = Path('/data/ldap/histopathologic/original_read_only/Lizard')
LABELS_DIR = Path('labels/Labels')
LABELS = ['Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Eosinophil', 'Connective tissue']


def imread(image_path):
    with Image.open(image_path) as image:
        # noinspection PyTypeChecker
        return np.array(image)


class Snapshot:
    """
    A Snapshot stands for a subregion in an image. It saves positional information and label information, but no image
    data.
    - The sample name: if the image file is named "images1/consep_10.png" the sample name is consep_10.
    - The image directory: either images1 or images2
    - The position of the subimage inside the image
    - A ndarray of size [N, 4] containing bounding boxes inside this subimage
    - A ndarray of size [N] containing class labels corresponding to the bounding boxes
    """
    def __init__(
        self, sample_name: str, image_directory: str, position: np.ndarray, bounding_boxes: np.ndarray,
        class_labels: np.ndarray
    ):
        self.sample_name = sample_name
        self.image_directory = Path(image_directory)
        self.position = position
        self.bounding_boxes: np.ndarray = bounding_boxes
        self.class_labels: np.ndarray = class_labels

    def get_image_path(self) -> Path:
        return self.image_directory / f'{self.sample_name}.png'

    def get_label_path(self) -> Path:
        return LABELS_DIR / f'{self.sample_name}.mat'

    def __repr__(self):
        return f'Snapshot(sample_name={self.sample_name} position=(y={self.position[0]}, x={self.position[1]})'


class LizardDetectionDataset(Dataset):
    def __init__(self, snapshots: List[Snapshot], data_dir: Path, image_size: np.ndarray, image_cache=None):
        """
        Args:
            snapshots: List of snapshots for this dataset
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_cache: The image cache to use. If None no image caching will be used.
        """
        self.snapshots = snapshots
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_cache: Dict[Path, np.ndarray] or None = image_cache

    @staticmethod
    def from_datadir(
        data_dir: Path, image_size: np.ndarray, image_stride: np.ndarray or None = None, use_cache: bool = False
    ):
        """
        Args:
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
            use_cache: Whether to keep loaded images in memory. Defaults to False.
        """
        # use image size as image stride, if no images stride provided
        image_stride = image_size if image_stride is None else image_stride
        snapshots = LizardDetectionDataset._define_snapshots(data_dir, image_size, image_stride)
        return LizardDetectionDataset(
            snapshots=snapshots,
            data_dir=data_dir,
            image_size=image_size,
            image_cache={} if use_cache else None,
        )

    @staticmethod
    def from_avocado(image_size: np.ndarray, image_stride: np.ndarray or None = None, use_cache: bool = False):
        """
        Args:
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
            use_cache: Whether to keep loaded images in memory. Defaults to False.
        """
        return LizardDetectionDataset.from_datadir(
            data_dir=AVOCADO_DATASET_LOCATION,
            image_size=image_size,
            image_stride=image_stride,
            use_cache=use_cache,
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

        # TODO: implement split
        subimages = []
        for image_file in image_files:
            subimages.extend(
                LizardDetectionDataset._snapshots_from_image_file(data_dir, image_file, image_size, image_stride)
            )

        return subimages

    @staticmethod
    def _snapshots_from_image_file(
            data_dir: Path, filename: Path, image_size: np.ndarray, image_stride: np.ndarray
    ) -> List[Snapshot]:
        with Image.open(filename) as image:
            width, height = image.size
        full_image_size = np.array([height, width])
        sample_name = filename.stem
        image_dir = filename.parent.stem
        position = np.array([0, 0])  # The position of the current snapshot as (y, x)
        snapshots = []

        all_label_data = LizardDetectionDataset._load_all_label_data(data_dir, sample_name)

        # iterate as long as right-bottom corner of subimage is in bounds of full_image_size
        while (position + image_size <= full_image_size).all():
            # filter relevant label data
            bounding_boxes = []
            class_labels = []
            for label, bounding_box in all_label_data:
                if LizardDetectionDataset._bounding_box_in_snapshot(position, image_size, bounding_box):
                    transformed_bounding_box = LizardDetectionDataset._transform_bounding_box(bounding_box, position)
                    bounding_boxes.append(transformed_bounding_box)
                    class_labels.append(label)

            bounding_boxes = np.stack(bounding_boxes) if bounding_boxes else np.array([])
            class_labels = np.stack(class_labels) if class_labels else np.array([])
            snapshots.append(Snapshot(sample_name, image_dir, position.copy(), bounding_boxes, class_labels))

            # move subimage to the right
            position[1] += image_stride[1]
            if position[1] + image_size[1] > full_image_size[1]:
                # move to next line
                position[1] = 0
                position[0] += image_stride[0]
        return snapshots

    def split(self, split_ratio: float, seed: int = 42) -> Tuple[Any, Any]:
        """
        Splits this dataset in two. The two split datasets do not share images.

        Args:
            split_ratio: A float between 0.0 and 1.0 defining the share of samples the first returned dataset includes.
        Returns:
            A tuple of two datasets.
        """
        snapshots_dict = {}  # maps sample_names to snapshots
        for snapshot in self.snapshots:
            if snapshot.sample_name not in snapshots_dict:
                snapshots_dict[snapshot.sample_name] = []
            snapshots_dict[snapshot.sample_name].append(snapshot)
        split_index = int(len(self.snapshots) * split_ratio)
        first_set = []
        second_set = []
        snapshots_lists_sorted = sorted(snapshots_dict.values(), key=lambda sl: sl[0].sample_name)
        random.Random(seed).shuffle(snapshots_lists_sorted)
        for snapshot_list in snapshots_lists_sorted:
            if len(first_set) >= split_index:
                second_set.extend(snapshot_list)
            else:
                first_set.extend(snapshot_list)
        return (
            LizardDetectionDataset(
                snapshots=first_set,
                data_dir=self.data_dir,
                image_size=self.image_size,
                image_cache=self.image_cache,
            ),
            LizardDetectionDataset(
                snapshots=second_set,
                data_dir=self.data_dir,
                image_size=self.image_size,
                image_cache=self.image_cache,
            ),
        )

    @staticmethod
    def _load_all_label_data(data_dir: Path, sample_name: str):
        label_path = data_dir / LABELS_DIR / '{}.mat'.format(sample_name)
        label_data = sio.loadmat(str(label_path))

        inst_map = label_data['inst_map']
        # noinspection PyTypeChecker
        nuclei_id: list = np.squeeze(label_data['id']).tolist()
        unique_values = np.unique(inst_map).tolist()[1:]  # remove 0 from begin; I checked and 0 is always at begin

        all_label_data = []
        for value in unique_values:
            idx = nuclei_id.index(value)
            # bounding box
            bounding_box = label_data['bbox'][idx]
            instance_class = label_data['class'][idx][0]  # checked. Is always a list with exactly one element.
            all_label_data.append((instance_class, bounding_box))
        return all_label_data

    def read_image(self, image_path):
        if self.image_cache is not None:
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
    def _transform_bounding_box(bounding_box, position) -> np.ndarray or None:
        """
        Args:
            bounding_box: The bounding box to transform as (y1, y2, x1, x2)
            position: The position of a snapshot
        Returns:
            The bounding box in form (y1, x1, y2, x2) where x and y coordinates are subtracted with the snapshot
            position.
        """
        return np.array([
            bounding_box[0] - position[0],  # y1
            bounding_box[2] - position[1],  # x1
            bounding_box[1] - position[0],  # y2
            bounding_box[3] - position[1],  # x2
        ])

    @staticmethod
    def _bounding_box_in_snapshot(position: np.ndarray, image_size: np.ndarray, bounding_box: np.ndarray) -> bool:
        # check y lower bound
        if bounding_box[0] < position[0] or bounding_box[1] < position[0]:
            return False
        # check x lower bound
        if bounding_box[2] < position[1] or bounding_box[3] < position[1]:
            return False
        upper_corner = position + image_size
        # check y upper bound
        if bounding_box[0] >= upper_corner[0] or bounding_box[1] >= upper_corner[0]:
            return False
        # check x upper bound
        if bounding_box[2] >= upper_corner[1] or bounding_box[3] >= upper_corner[1]:
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
        return {
            'image': subimage,
            'boxes': snapshot.bounding_boxes,
            'labels': snapshot.class_labels,
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
