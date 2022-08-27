"""
Lizard Dataset described here: https://arxiv.org/pdf/2108.11195.pdf

TODO: rework transform bounding box
"""
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable

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
        show_progress: bool = False
    ):
        """
        Args:
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
            use_cache: Whether to keep loaded images in memory. Defaults to False.
            show_progress: Whether to show loading progress with tqdm
        """
        # use image size as image stride, if no images stride provided
        image_stride = image_size if image_stride is None else image_stride
        snapshots = LizardDetectionDataset._define_snapshots(data_dir, image_size, image_stride, show_progress)
        max_boxes_per_snapshot = 0
        for snapshot in snapshots:
            max_boxes_per_snapshot = max(snapshot.bounding_boxes.shape[0], max_boxes_per_snapshot)
        return LizardDetectionDataset(
            snapshots=snapshots,
            data_dir=data_dir,
            image_size=image_size,
            max_boxes_per_snapshot=max_boxes_per_snapshot,
            image_cache={} if use_cache else None,
        )

    @staticmethod
    def from_avocado(
            image_size: np.ndarray, image_stride: np.ndarray or None = None, use_cache: bool = False,
            show_progress: bool = False, force_one_class: bool = False,
    ):
        """
        Args:
            image_size: The size of the images returned by __getitem__ as [height, width]
            image_stride: The stride between the images returned by __getitem__ as [y, x]. Defaults to <image_size>
            use_cache: Whether to keep loaded images in memory. Defaults to False.
            force_one_class: If set to True, will always give class 0 as class label
        """
        return LizardDetectionDataset.from_datadir(
            data_dir=AVOCADO_DATASET_LOCATION,
            image_size=image_size,
            image_stride=image_stride,
            use_cache=use_cache,
            show_progress=show_progress,
            force_one_class=force_one_class,
        )

    @staticmethod
    def _define_snapshots(
            data_dir: Path, image_size: np.ndarray, image_stride: np.ndarray, show_progress: bool
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
                    class_labels.append(label - 1)  # labels to 0 - 5

            if bounding_boxes:
                bounding_boxes = np.array(bounding_boxes, dtype=np.float32)
                assert image_size[0] == image_size[1]
                bounding_boxes /= float(image_size[0])  # scale to relative coordinates
                class_labels = np.array(class_labels)
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
                max_boxes_per_snapshot=self.max_boxes_per_snapshot,
                image_cache=self.image_cache,
            ),
            LizardDetectionDataset(
                snapshots=second_set,
                data_dir=self.data_dir,
                image_size=self.image_size,
                max_boxes_per_snapshot=self.max_boxes_per_snapshot,
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
    def _transform_bounding_box(bounding_box, position) -> np.ndarray or None:
        """
        Args:
            bounding_box: The bounding box to transform as (y1, y2, x1, x2)
            position: The position of a snapshot
        Returns:
            The bounding box in form (y1, x1, y2, x2) where x and y coordinates are subtracted with the snapshot
            position. The resulting bounding box is in tlbr-format.
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

    def pad_join_boxes_and_labels(self, bounding_boxes: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
        """
        Converts given bounding_boxes from tlbr to ltrb format.
        Then joins the given bounding_boxes with shape [N, 4] and class_labels with shape [N,] to a new tensor with
        shape [N, 5] so each sample has elements (class_label, left, top, right, bottom).
        Also pads with (-1, 0, 0, 0, 0) samples to create shape of [max_boxes_per_snapshot, 5].

        :param bounding_boxes: Bounding boxes of shape [N, 4]
        :param class_labels: Class labels of shape [N,]
        :return: Padded and joint bounding boxes and labels with shape [max_boxes_per_snapshot, 5]
        """
        assert class_labels.shape[0] > 0
        assert class_labels.shape[0] == bounding_boxes.shape[0]
        assert (class_labels >= 0).all()
        assert (class_labels < len(LABELS)).all()

        if self.force_one_class:
            class_labels = 0  # always state class 0

        # convert from tlbr to ltrb
        indices = torch.LongTensor([1, 0, 3, 2])
        bounding_boxes = bounding_boxes[:, indices]

        # join labels and boxes
        joint = np.concatenate((class_labels.reshape(-1, 1), bounding_boxes), axis=1)

        # pad
        add = self.max_boxes_per_snapshot - joint.shape[0]
        pad = np.zeros((add, 5), dtype=np.float32)
        pad[:, 0] = -1
        return np.concatenate((joint, pad), axis=0, dtype=np.float32)

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
        labeled_boxes = self.pad_join_boxes_and_labels(snapshot.bounding_boxes, snapshot.class_labels)
        return {
            'image': self.to_tensor(subimage),
            'boxes': labeled_boxes,
        }

    def __len__(self):
        return len(self.snapshots)
