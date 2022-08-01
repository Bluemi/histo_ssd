from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from d2l.torch import d2l
from determined.pytorch import DataLoader

from datasets import LizardDetectionDataset
from models import TinySSD
from utils.bounding_boxes import multibox_target, multibox_detection
from utils.funcs import draw_boxes, show_image, debug


def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor):
    # Because the class prediction results are on the final dimension, `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def predict(images: torch.Tensor, net: TinySSD, device: str = 'cpu'):
    """
    Predict batches of images

    :param images: The images to predict the net on with shape [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS]
    :param net: The network to use.
    :param device: The device to compute on
    :return:
    """
    anchors, cls_preds, bbox_preds = net(images.to(device))
    cls_probs = functional.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]  # todo: 0 means only the first sample of the batch is used


def main():
    device = 'cpu'
    net = TinySSD(num_classes=6)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bbox_loss = torch.nn.L1Loss(reduction='none')

    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([224, 224]),
        image_stride=np.array([224, 224]),
        use_cache=True,
        show_progress=True,
    )

    # train_dataset, eval_dataset = dataset.split(0.5)
    train_dataset = dataset

    print(f'len train: {len(train_dataset)}')
    # print(f'len eval: {len(eval_dataset)}')

    train_data_loader = DataLoader(train_dataset, batch_size=32)
    eval_data_loader = DataLoader(train_dataset, batch_size=32)

    def calc_loss(class_preds, class_labels, bounding_box_preds, bounding_box_labels, bounding_box_masks):
        batch_size, num_classes = class_preds.shape[0], class_preds.shape[2]
        cls = cls_loss(class_preds.reshape(-1, num_classes), class_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(bounding_box_preds * bounding_box_masks, bounding_box_labels * bounding_box_masks).mean(dim=1)
        return cls + bbox

    num_epochs = 30
    timer = d2l.Timer()

    net = net.to(device)
    for epoch in range(num_epochs):
        print(f'--- epoch {epoch} ---')
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        net.train()

        metrics = [0.0] * 4

        for batch in train_data_loader:
            timer.start()
            features = batch['image']
            target = batch['boxes']
            optimizer.zero_grad()
            x, y = features.to(device), target.to(device)
            # Generate multiscale anchor boxes and predict their classes and offsets
            anchors, cls_preds, bbox_preds = net(x)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y)
            # Calculate the loss function using the predicted and labeled values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            optimizer.step()
            metrics[0] += cls_eval(cls_preds, cls_labels)
            metrics[1] += cls_labels.numel()
            metrics[2] += bbox_eval(bbox_preds, bbox_labels, bbox_masks),
            metrics[3] += bbox_labels.numel()
            print('loss: {}'.format(l.mean()))
            print('class eval: {}'.format(cls_eval(cls_preds, cls_labels)))
            print('bbox eval: {}'.format(bbox_eval(bbox_preds, bbox_labels, bbox_masks)))
        cls_err, bbox_mae = 1 - metrics[0] / metrics[1], metrics[2] / metrics[3]
        print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
        print(f'{len(train_dataset) / timer.stop():.1f} examples/sec on {str(device)}')

    print('---')
    threshold = 0.15

    for batch in eval_data_loader:
        net.eval()
        for img, boxes in zip(batch['image'], batch['boxes']):
            boxes = boxes[:, 1:]
            pred_img = img.unsqueeze(0)
            output = predict(pred_img, net)
            draw_img = img.permute(1, 2, 0)
            draw_img_orig = draw_img.clone()
            draw_boxes(draw_img_orig, boxes)
            show_image(draw_img_orig)
            for row in output:
                if row[1] > threshold:
                    draw_boxes(draw_img, row[None, 2:])
            show_image(draw_img)

    # print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')


if __name__ == '__main__':
    main()
