from pathlib import Path

import numpy as np
import torch
from determined.pytorch import DataLoader

from datasets import LizardDetectionDataset
from models import TinySSD
from utils.bounding_boxes import multibox_target


def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor):
    # Because the class prediction results are on the final dimension, `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


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

    data_loader = DataLoader(dataset, batch_size=8)

    def calc_loss(class_preds, class_labels, bounding_box_preds, bounding_box_labels, bounding_box_masks):
        batch_size, num_classes = class_preds.shape[0], class_preds.shape[2]
        cls = cls_loss(class_preds.reshape(-1, num_classes), class_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(bounding_box_preds * bounding_box_masks, bounding_box_labels * bounding_box_masks).mean(dim=1)
        return cls + bbox

    num_epochs = 20

    net = net.to(device)
    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        net.train()
        for batch in data_loader:
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
            print('loss: {}'.format(l.mean()))
            print('class eval: {}'.format(cls_eval(cls_preds, cls_labels)))
            print('bbox eval: {}'.format(bbox_eval(bbox_preds, bbox_labels, bbox_masks)))
        # cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    # print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')


if __name__ == '__main__':
    main()
