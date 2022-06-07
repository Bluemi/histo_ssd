import torch
from skimage.io import imread


SIMPLE_SCALES = True
torch.set_printoptions(2)


def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-3:-1]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((
        size_tensor * torch.sqrt(ratio_tensor[0]),
        sizes[0] * torch.sqrt(ratio_tensor[1:])
    )) * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((
        size_tensor / torch.sqrt(ratio_tensor[0]),
        sizes[0] / torch.sqrt(ratio_tensor[1:]))
    )
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack(
        [shift_x, shift_y, shift_x, shift_y], dim=1
    ).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def create_anchor_boxes(image_shape, scales, ratios):
    # boxes_per_pixel = len(scales) + len(ratios) - 1
    # anchor_shape = (*image_shape, boxes_per_pixel, 4)

    y_positions = torch.arange(0, image_shape[0], dtype=torch.float32)
    x_positions = torch.arange(0, image_shape[1], dtype=torch.float32)

    center_points = torch.dstack(torch.meshgrid(y_positions, x_positions, indexing='ij'))

    # normalize center points
    if image_shape[0] == image_shape[1]:
        center_points = center_points / image_shape[0]
    else:
        center_points[:, :, 0] /= float(image_shape[0])
        center_points[:, :, 1] /= float(image_shape[1])
    print(center_points)
    print(center_points.shape)


def main():
    data = torch.tensor(imread('res/black_rect.png'))
    if SIMPLE_SCALES:
        scales = [1.0]
        ratios = [1.0]
    else:
        scales = [0.25, 0.5, 0.75]
        ratios = [0.5, 1, 2]
    # boxes = multibox_prior(data, sizes=scales, ratios=ratios)
    create_anchor_boxes(data.shape[-3:-1], scales=scales, ratios=ratios)

    # boxes = boxes.reshape(8, 5, 1, 4)
    # print(boxes.size())
    # print(boxes[0, 0, 0, :])


if __name__ == '__main__':
    main()
