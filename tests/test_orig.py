import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import LizardDetectionDataset
from datasets.banana_dataset import load_data_bananas
from utils.bounding_boxes import create_anchor_boxes, multibox_target, multibox_detection
from utils.funcs import draw_boxes


DISPLAY_GROUND_TRUTH = True
DATASET = 'banana'
DATASET = 'lizard'

MODEL_LOAD_PATH = '../models/{}_model2.pth'.format(DATASET)

if DATASET == 'banana':
    NUM_CLASSES = 1
elif DATASET == 'lizard':
    NUM_CLASSES = 6
else:
    raise ValueError('Unknown dataset: {}'.format(DATASET))


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(x, blk, size, ratio, cls_predictor, bbox_predictor):
    y = blk(x)
    anchors = create_anchor_boxes(y.shape[-2:], scales=size, ratios=ratio, device=y.device)
    cls_preds = cls_predictor(y)
    bbox_preds = bbox_predictor(y)
    return y, anchors, cls_preds, bbox_preds


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    def __init__(self, num_classes):
        super(TinySSD, self).__init__()
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


batch_size = 32
if DATASET == 'banana':
    train_iter, val_iter = load_data_bananas('../data/banana-detection', batch_size)
elif DATASET == 'lizard':
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([224, 224]),
        image_stride=np.array([224, 224]),
        use_cache=True,
        show_progress=False,
    )
    train_dataset, val_dataset = dataset.split(0.8)
    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_iter = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
else:
    raise ValueError('Unknown dataset: {}'.format(DATASET))


device = torch.device('cpu')
net = TinySSD(num_classes=NUM_CLASSES)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor):
    # Because the class prediction results are on the final dimension, `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


num_epochs = 20
net = net.to(device)

if MODEL_LOAD_PATH and os.path.isfile(MODEL_LOAD_PATH):
    net.load_state_dict(torch.load(MODEL_LOAD_PATH))
else:
    cls_err = None
    bbox_mae = None
    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy, Sum of absolute error, no. of examples
        # in sum of absolute error
        metric = [0.0] * 4
        net.train()
        for batch in tqdm(train_iter, desc=f'epoch {epoch + 1}'):
            features = batch['image']
            target = batch['boxes']
            trainer.zero_grad()
            x, Y = features.to(device), target.to(device)
            # Generate multiscale anchor boxes and predict their classes and offsets
            anchors, cls_preds, bbox_preds = net(x)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # Calculate the loss function using the predicted and labeled values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric[0] += cls_eval(cls_preds, cls_labels)
            metric[1] += cls_labels.numel()
            metric[2] += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            metric[3] += bbox_labels.numel()
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset)} examples on {str(device)}')
    torch.save(net.state_dict(), MODEL_LOAD_PATH)


# Prediction

def predict(x):
    net.eval()
    anchors, cls_preds, bbox_preds = net(x.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


for batch in val_iter:
    images = batch['image']
    boxes = batch['boxes']
    for image, box in zip(images, boxes):
        if DATASET == 'lizard':
            draw_image = image * 255
        else:
            draw_image = image
        draw_image = draw_image.squeeze(0).permute(1, 2, 0).long()
        output = predict(image.unsqueeze(0).float())

        def display(img, output, boxes, threshold):
            for row in output:
                score = float(row[1])
                if score < threshold:
                    continue
                bbox = row[2:6].unsqueeze(0)
                draw_boxes(img, bbox, color=(255, 0, 0), box_format='ltrb')
            if DISPLAY_GROUND_TRUTH:
                for box in boxes:
                    if box[0] < 0:
                        continue
                    bbox = box[1:5].unsqueeze(0)
                    draw_boxes(img, bbox, color=(0, 255, 0), box_format='ltrb')
            plt.imshow(img)
            plt.show()

        display(draw_image, output.cpu(), box, threshold=0.9)
