import inspect
import sys
from typing import Union, List, NewType, Tuple, Optional

import torch
from matplotlib import pyplot as plt

Color = NewType('Color', Tuple[int, int, int])
DEFAULT_COLORS1 = torch.tensor([
    [0, 255, 0],      # green
    [255, 0, 0],      # red
    [0, 0, 255],      # blue
    [255, 255, 0],    # yellow
    [0, 255, 255],    # cyan
    [220, 220, 220],  # gray
])

DARK_COLORS = torch.tensor([
    [0, 140, 0],      # green
    [170, 0, 0],      # red
    [0, 0, 150],      # blue
    [180, 180, 0],    # yellow
    [0, 170, 170],    # cyan
    [120, 120, 120],  # gray
])

BRIGHT_COLORS = torch.tensor([
    [30, 255, 30],    # green
    [255, 30, 30],    # red
    [0, 80, 255],    # blue
    [255, 255, 50],   # yellow
    [70, 255, 255],   # cyan
    [220, 220, 220],  # gray
])


def show_image(image):
    plt.imshow(image)
    plt.show()


def draw_boxes(
        image: torch.Tensor, bounding_boxes: torch.Tensor,
        color: Union[Tuple[int, int, int], List[Tuple[int, int, int]], int, str, torch.Tensor, None] = 'random',
        color_indices: Optional[torch.Tensor] = None, box_format: str = 'tlbr', sign: str = 'box', color_mode='set',
):
    """
    Draws the given bounding boxes into the given image.
    :param image: The image to draw in
    :param bounding_boxes: The bounding boxes to draw. A tensor of shape (nBoxes, 4).
                           See box_format for more information.
    :param color: The colors to use. If None, black is used. Uses cycling for more boxes than colors.
    :param color_indices: Tensor with shape (nBoxes,). Gives the color index for each box.
    :param box_format: The format of the bounding box. Either tlbr (top, left, bottom, right) or
                       ltrb (left, top, right, bottom).
    :param sign: Which sign to draw. One of ['box', 'cross']. Defaults to box.
    :param color_mode: One of ['set', 'add']. If set, colors are set, if add colors are added.
    """
    def _draw_box(img, l, t, r, bot_arg, col):
        lines = [
            img[t, l:r],  # draw top line
            img[bot_arg, l:r],  # draw bottom line
            img[t:bot_arg, l],  # draw left line
            img[t:bot_arg, r],  # draw right line
        ]
        for line in lines:
            if color_mode == 'set':
                line[:] = torch.clamp(col, torch.tensor(0), torch.tensor(255))
            elif color_mode == 'add':
                line[:] = torch.clamp(line + col, torch.tensor(0), torch.tensor(255))

    def _draw_cross(img, l, t, r, b, col):
        cross_size = 5
        center_x = torch.div((l + r), 2, rounding_mode='floor')
        center_y = torch.div((t + b), 2, rounding_mode='floor')

        # draw horizontal line
        img[center_y, center_x-cross_size:center_x+cross_size] = col
        # draw vertical line
        img[center_y-cross_size:center_y+cross_size, center_x] = col

    if sign == 'box':
        draw_function = _draw_box
    elif sign == 'cross':
        draw_function = _draw_cross
    else:
        raise ValueError('Unknown sign: "{}"'.format(sign))

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

    if isinstance(color, torch.Tensor):
        pass
    elif not color:
        color = [0.0]
    elif color == 'random':
        pass
    elif not isinstance(color, list):
        color = [color]

    if color_indices is not None:
        if color_indices.dtype != torch.int32:
            color_indices = color_indices.to(torch.int32)

    for index, box in enumerate(bounding_boxes):
        if color_indices is not None:
            index = color_indices[index]
        if isinstance(color, list) or isinstance(color, torch.Tensor):
            c = color[index % len(color)]
        else:
            c = (torch.rand(3) * 255).to(torch.uint8)

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
        draw_function(image, left, top, right, bot, c)


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
