"""
Lizard Dataset described here: https://arxiv.org/pdf/2108.11195.pdf

TODO: rework transform bounding box
"""
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable, Optional

import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


AVOCADO_DATASET_LOCATION = Path('/data/ldap/histopathologic/original_read_only/Lizard')
LABELS_DIR = Path('labels/Labels')
LABELS = ['Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Eosinophil', 'Connective tissue']


def imread(image_path):
    with Image.open(image_path) as image:
        # noinspection PyTypeChecker
        return np.array(image)


def get_progress_func(show_progress: bool) -> Callable:
    if show_progress:
        progress_function = tqdm
    else:
        def progress_function(x, **_kwargs):
            return x
    return progress_function


def _boxes_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class Snapshot:
    """
    A Snapshot stands for a subregion in an image. It saves positional information and label information, but no image
    data.
    - The sample name: if the image file is named "images1/consep_10.png" the sample name is consep_10.
    - The image directory: either images1 or images2
    - The position of the subimage inside the image
    - A ndarray of size [N, 5] containing class labels and bounding boxes inside this subimage
    - A ndarray of size [N, 4] containing bounding boxes inside this subimage
    - A ndarray of size [N] containing class labels corresponding to the bounding boxes
    """
    def __init__(
        self, sample_name: str, image_directory: str, position: np.ndarray, label_data: np.ndarray,
    ):
        self.sample_name = sample_name
        self.image_directory = Path(image_directory)
        self.position = position
        self.label_data: np.ndarray = label_data

    def get_image_path(self) -> Path:
        return self.image_directory / f'{self.sample_name}.png'

    def get_label_path(self) -> Path:
        return LABELS_DIR / f'{self.sample_name}.mat'

    def __repr__(self):
        return f'Snapshot(sample_name={self.sample_name} position=(y={self.position[0]}, x={self.position[1]})'


class LizardDetectionDataset(Dataset):
    def __init__(
            self, snapshots: List[Snapshot], data_dir: Path, image_size: np.ndarray, max_boxes_per_snapshot: int,
            image_cache=None, force_one_class: bool = False,
    ):
        """
        Args:
            snapshots: List of snapshots for this dataset
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            max_boxes_per_snapshot: The maximum number of bounding boxes per snapshot
            image_cache: The image cache to use. If None no image caching will be used.
            force_one_class: Always return class 0 as label
        """
        self.snapshots = snapshots
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_cache: Dict[Path, np.ndarray] or None = image_cache
        self.max_boxes_per_snapshot = max_boxes_per_snapshot
        self.to_tensor = transforms.ToTensor()
        self.force_one_class = force_one_class

    @staticmethod
    def from_datadir(
        data_dir: Path, image_size: np.ndarray, image_stride: np.ndarray or None = None, use_cache: bool = False,
        show_progress: bool = False, force_one_class: bool = False, ignore_classes: Optional[List[int]] = None
    ):
        """
        Args:
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
            use_cache: Whether to keep loaded images in memory. Defaults to False.
            show_progress: Whether to show loading progress with tqdm
            force_one_class: Always return class 0 as label
            ignore_classes: A list of class indices that should not be suppressed in the output of the dataset.
        """
        assert not (force_one_class and ignore_classes), 'Options force_one_class and ignore_classes are incompatible'
        if ignore_classes is None:
            ignore_classes = []
        # use image size as image stride, if no images stride provided
        image_stride = image_size if image_stride is None else image_stride
        snapshots = LizardDetectionDataset._define_snapshots(
            data_dir, image_size, image_stride, show_progress, ignore_classes
        )
        max_boxes_per_snapshot = 0
        for snapshot in snapshots:
            max_boxes_per_snapshot = max(snapshot.label_data.shape[0], max_boxes_per_snapshot)

        return LizardDetectionDataset(
            snapshots=snapshots,
            data_dir=data_dir,
            image_size=image_size,
            max_boxes_per_snapshot=max_boxes_per_snapshot,
            image_cache={} if use_cache else None,
            force_one_class=force_one_class,
        )

    @staticmethod
    def from_avocado(
            image_size: np.ndarray, image_stride: np.ndarray or None = None, use_cache: bool = False,
            show_progress: bool = False, force_one_class: bool = False, ignore_classes: Optional[List[int]] = None
    ):
        """
        Args:
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
            use_cache: Whether to keep loaded images in memory. Defaults to False.
            force_one_class: If set to True, will always give class 0 as class label
            ignore_classes: A list of class indices that should not be suppressed in the output of the dataset.
        """
        return LizardDetectionDataset.from_datadir(
            data_dir=AVOCADO_DATASET_LOCATION,
            image_size=image_size,
            image_stride=image_stride,
            use_cache=use_cache,
            show_progress=show_progress,
            force_one_class=force_one_class,
            ignore_classes=ignore_classes,
        )

    @staticmethod
    def _define_snapshots(
            data_dir: Path, image_size: np.ndarray, image_stride: np.ndarray, show_progress: bool, ignore_classes,
    ) -> List[Snapshot]:
        """
        Returns a list of snapshots to use for this dataset. Images will be searched in '<data_dir>/images1/*.png' and
        '<data_dir>/images2/*.png'.
        """
        image1_files = sorted([f for f in (data_dir / 'images1').iterdir() if str(f).endswith('.png')])
        image2_files = sorted([f for f in (data_dir / 'images2').iterdir() if str(f).endswith('.png')])
        image_files = image1_files + image2_files

        subimages = []
        progress_function = get_progress_func(show_progress)
        for image_file in progress_function(image_files, desc='Loading Dataset: '):
            subimages.extend(
                LizardDetectionDataset._snapshots_from_image_file(
                    data_dir, image_file, image_size, image_stride, ignore_classes
                )
            )

        return subimages

    @staticmethod
    def _filter_and_transform_all_label_data(all_label_data: np.ndarray, position, image_size) -> np.ndarray:
        # NOTE: position is (y, x), but all_label_data is (c, x, y, x, y)
        bot_right = position + image_size

        # bbox top >= image top
        included_indices = all_label_data[:, 2] >= position[0]

        # bbox bot < image bot
        included_indices = np.logical_and(included_indices, all_label_data[:, 4] < bot_right[0])

        # bbox left >= image left
        included_indices = np.logical_and(included_indices, all_label_data[:, 1] >= position[1])

        # bbox right < image right
        included_indices = np.logical_and(included_indices, all_label_data[:, 3] < bot_right[1])

        image_data = all_label_data[included_indices].astype(np.float32)

        # subtract position to move to center
        image_data[:, (1, 3)] -= position[1]
        image_data[:, (2, 4)] -= position[0]

        # format to relative
        image_data[:, (1, 3)] /= float(image_size[1])
        image_data[:, (2, 4)] /= float(image_size[0])

        return image_data

    @staticmethod
    def _remap_ignored_classes(all_label_data, ignore_classes):
        # maps class labels to new class labels
        label_map = list(range(len(LABELS) - len(ignore_classes)))
        for ignore_class in sorted(ignore_classes, reverse=False):
            label_map.insert(ignore_class, None)

        # remove ignored classes
        for old_label, new_label in enumerate(label_map):
            if new_label is None:  # remove class
                all_label_data = all_label_data[all_label_data[:, 0] != old_label]
            else:
                old_label_indices = all_label_data[:, 0] == old_label
                all_label_data[old_label_indices, 0] = new_label  # map to new label

        return all_label_data

    @staticmethod
    def _snapshots_from_image_file(
            data_dir: Path, filename: Path, image_size: np.ndarray, image_stride: np.ndarray, ignore_classes
    ) -> List[Snapshot]:
        with Image.open(filename) as image:
            width, height = image.size
        full_image_size = np.array([height, width])
        sample_name = filename.stem
        image_dir = filename.parent.stem
        position = np.array([0, 0])  # The position of the current snapshot as (y, x)
        snapshots = []

        all_label_data = LizardDetectionDataset._load_all_label_data(data_dir, sample_name)

        # maps class labels to new class labels
        if ignore_classes:
            all_label_data = LizardDetectionDataset._remap_ignored_classes(all_label_data, ignore_classes)

        # iterate as long as right-bottom corner of subimage is in bounds of full_image_size
        while (position + image_size <= full_image_size).all():
            image_label_data = LizardDetectionDataset._filter_and_transform_all_label_data(
                all_label_data, position, image_size
            )

            if image_label_data.shape[0] > 0:
                snapshots.append(Snapshot(sample_name, image_dir, position.copy(), image_label_data))

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
                max_boxes_per_snapshot=self.max_boxes_per_snapshot,
                image_cache=self.image_cache,
                force_one_class=self.force_one_class,
            ),
            LizardDetectionDataset(
                snapshots=second_set,
                data_dir=self.data_dir,
                image_size=self.image_size,
                max_boxes_per_snapshot=self.max_boxes_per_snapshot,
                image_cache=self.image_cache,
                force_one_class=self.force_one_class,
            ),
        )

    @staticmethod
    def _load_all_label_data(data_dir: Path, sample_name: str):
        """
        Returns all label data of an image with shape (NUM_SAMPLES, 5) in form (classlabel, min_y, min_x, max_y, max_x)

        :param data_dir: The data directory to load labels from
        :param sample_name: The name of the sample to load
        :return: label data of the sample
        """
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
            assert instance_class >= 1
            all_label_data.append((instance_class - 1, *bounding_box))  # label -1 to have labels 0 - 5
        all_label_data = np.array(all_label_data)
        # reorder from (y_min, y_max, x_min, x_max) to (min_x, min_y, max_x, max_y)
        all_label_data = all_label_data[:, (0, 3, 1, 4, 2)]

        assert np.all(all_label_data[:, 1] < all_label_data[:, 3])
        assert np.all(all_label_data[:, 2] < all_label_data[:, 4])

        # sort out invalid boxes
        special_cases = ('glas_60', 'consep_3')
        areas = _boxes_area(all_label_data[:, 1:])
        area_threshold = (300 * 300 * 0.043)  # threshold found by testing, that excludes most invalid boxes
        if sample_name in special_cases:
            area_threshold = (300 * 300 * 0.03)
        valid = areas < area_threshold
        all_label_data = all_label_data[valid]

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

    def get_subimage(self, snapshot) -> torch.Tensor:
        image_path = self.data_dir / snapshot.get_image_path()
        image = self.read_image(image_path)
        return image[
            snapshot.position[0]:snapshot.position[0]+self.image_size[0],
            snapshot.position[1]:snapshot.position[1]+self.image_size[1],
        ]

    def get_label_data(self, snapshot) -> Dict:
        label_path = self.data_dir / snapshot.get_label_path()
        return sio.loadmat(str(label_path))

    @staticmethod
    def normalization_values():
        # Calculated on the 'train' set
        mean = [0.64788544, 0.4870253,  0.68022424]
        std = [0.2539682,  0.22869842, 0.24064516]
        return mean, std

    def pad_join_boxes_and_labels(self, label_data: np.ndarray) -> np.ndarray:
        """
        Converts given bounding_boxes from tlbr to ltrb format.
        Then joins the given bounding_boxes with shape [N, 4] and class_labels with shape [N,] to a new tensor with
        shape [N, 5] so each sample has elements (class_label, left, top, right, bottom).
        Also pads with (-1, 0, 0, 0, 0) samples to create shape of [max_boxes_per_snapshot, 5].

        :param bounding_boxes: Bounding boxes of shape [N, 4]
        :param class_labels: Class labels of shape [N,]
        :return: Padded and joint bounding boxes and labels with shape [max_boxes_per_snapshot, 5]
        """
        assert label_data[:, 0].shape[0] > 0
        assert (label_data[:, 0] >= 0).all()
        assert (label_data[:, 0] < len(LABELS)).all()

        if self.force_one_class:
            label_data[:, 0].fill(0)  # always state class 0

        # pad
        add = self.max_boxes_per_snapshot - label_data.shape[0]
        pad = np.zeros((add, 5), dtype=np.float32)
        pad[:, 0] = -1
        return np.concatenate((label_data, pad), axis=0, dtype=np.float32)

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Gets the dataset sample with the given index. Every sample contains N instances of nuclei.
        A sample is a dictionary with the following keys:
          - image: An image with shape [height, width, 3]
          - boxes: A List[N, 5] containing the bounding boxes with corresponding label.
                   Each sample as the form [class_label, top, left, bottom, right] with 0 <= class_label <= 5.
        """
        snapshot = self.snapshots[index]
        subimage = self.get_subimage(snapshot)
        labeled_boxes = self.pad_join_boxes_and_labels(snapshot.label_data)
        return {
            'image': self.to_tensor(subimage),
            'boxes': labeled_boxes,
            'sample': snapshot.sample_name,
        }

    def __len__(self):
        return len(self.snapshots)
