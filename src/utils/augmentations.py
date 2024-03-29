import random

import numpy as np
from torch import nn
from torchvision.transforms.functional import rotate, hflip


L = 1
T = 2
R = 3
B = 4


def _flip_rotate_by_indices(boxes, indices):
    boxes = boxes[:, np.abs(indices)]
    for i in range(1, 5):
        if indices[i] < 0:
            boxes[:, i] = 1.0 - boxes[:, i]
    return boxes


class RandomRotate(nn.Module):
    """
    Augmentation method to randomly rotate the given image and the given bounding boxes by 0, 90, 180 or 270 degrees.
    Also, randomly flips the image.
    """
    def __init__(self, angles=None):
        super().__init__()
        if angles is None:
            angles = [0, 90, 180, 270]
        self.angles = angles

    def forward(self, x):
        """
        Rotates the given image and the given bounding boxes by 0, 90, 180 or 270 degree.

        :param x: A dictionary with keys 'image' and 'boxes', where image is a tensor with shape (3, height, width)
                  and 'boxes' is a tensor with shape (NUM_BOXES, 5) where each element is
                  (class_label, left, top, right, bottom).
        :return: The same dictionary but with image and boxes rotated and flipped
        """
        assert isinstance(x, dict)
        image = x['image']
        boxes = x['boxes']

        angle = random.choice(self.angles)

        if angle != 0:
            # rotate image
            image = rotate(image, angle)

            # rotate boxes
            if angle == 90:
                indices = [0, T, -R, B, -L]
            elif angle == 180:
                indices = [0, -R, -B, -L, -T]
            elif angle == 270:
                indices = [0, -B, L, -T, R]
            else:
                raise ValueError('cannot rotate by {} degrees'.format(angle))

            boxes = _flip_rotate_by_indices(boxes, indices)

        return {
            'image': image,
            'boxes': boxes,
        }


class RandomFlip(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        """
        Randomly flips the given image and bounding boxes horizontally.
        """
        assert isinstance(x, dict)
        image = x['image']
        boxes = x['boxes']

        flip = random.choice([True, False])
        if flip:
            image = hflip(image)
            indices = [0, -R, T, -L, B]
            boxes = _flip_rotate_by_indices(boxes, indices)

        return {
            'image': image,
            'boxes': boxes,
        }
