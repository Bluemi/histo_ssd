import inspect
import sys
from typing import Union, List, NewType, Tuple

import torch
from matplotlib import pyplot as plt

Color = NewType('Color', Tuple[int, int, int])


def show_image(image):
    plt.imshow(image)
    plt.show()


def draw_boxes(
        image: torch.Tensor, bounding_boxes: torch.Tensor,
        color: Union[Tuple[int, int, int], List[Tuple[int, int, int]], int, str, None] = 'random',
        box_format: str = 'tlbr'
):
    """
    Draws the given bounding boxes into the given image.
    :param image: The image to draw in
    :param bounding_boxes: The bounding boxes to draw. A tensor of shape (nBoxes, 4).
                           Each bounding box is (top, left, bottom, right).
    :param color: The colors to use. If None, black is used. Uses cycling for more boxes than colors.
    :param box_format: The format of the bounding box. Either tlbr or ltrb
    """
    bounding_boxes = bounding_boxes.clone()
    # scale box to int position
    if bounding_boxes.dtype == torch.float32:
        if box_format == 'tlbr':
            bounding_boxes[:, 0] *= image.shape[0]
            bounding_boxes[:, 2] *= image.shape[0]
            bounding_boxes[:, 1] *= image.shape[1]
            bounding_boxes[:, 3] *= image.shape[1]
            bounding_boxes = bounding_boxes.to(torch.int)
        elif box_format == 'ltrb':
            bounding_boxes[:, 0] *= image.shape[1]
            bounding_boxes[:, 2] *= image.shape[1]
            bounding_boxes[:, 1] *= image.shape[0]
            bounding_boxes[:, 3] *= image.shape[0]
            bounding_boxes = bounding_boxes.to(torch.int)
        else:
            raise ValueError('Unknown box format: {}'.format(box_format))

    # remove alpha channel
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, 0:3]

    if not color:
        color = [0.0]
    elif color == 'random':
        pass
    elif not isinstance(color, list):
        color = [color]

    for index, box in enumerate(bounding_boxes):
        if isinstance(color, list):
            c = color[index % len(color)]
        else:
            c = (torch.rand(3) * 255).to(torch.int8)

        if isinstance(c, tuple):
            c = torch.tensor(c)

        # normalize box points
        if box_format == 'tlbr':
            top = min(max(box[0], 0), image.shape[0]-1)
            bot = min(max(box[2], 0), image.shape[0]-1)
            left = min(max(box[1], 0), image.shape[1]-1)
            right = min(max(box[3], 0), image.shape[1]-1)
        elif box_format == 'ltrb':
            top = min(max(box[1], 0), image.shape[0]-1)
            bot = min(max(box[3], 0), image.shape[0]-1)
            left = min(max(box[0], 0), image.shape[1]-1)
            right = min(max(box[2], 0), image.shape[1]-1)
        else:
            raise ValueError('Unknown box format: {}'.format(box_format))

        # draw box
        image[top, left:right] = c  # draw top line
        image[bot, left:right] = c  # draw bottom line
        image[top:bot, left] = c  # draw left line
        image[top:bot, right] = c  # draw right line


def debug(arg):
    """
    Print name of arg and arg.
    :param arg: The argument to print
    """
    # noinspection PyUnresolvedReferences,PyProtectedMember
    fr = sys._getframe(1)  # type: frame
    code = inspect.getsource(fr).split('\n')
    line = code[fr.f_lineno - fr.f_code.co_firstlineno]
    varname = line.partition('debug(')[2].rpartition(')')[0]
    print('{}: {}'.format(varname, arg))
