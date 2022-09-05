import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

from datasets import LizardDetectionDataset
from datasets.banana_dataset import load_data_bananas
from utils.clock import Clock
from utils.metrics import update_mean_average_precision, calc_loss, cls_eval, bbox_eval
from models import SSDModel, predict
from utils.bounding_boxes import multibox_target
from utils.funcs import draw_boxes, debug

DISPLAY_GROUND_TRUTH = True
DATASET = 'banana'
DATASET = 'lizard'

MODEL_LOAD_PATH = '../models/{}_model2.pth'.format(DATASET)
# MODEL_LOAD_PATH = None

if DATASET == 'banana':
    NUM_CLASSES = 1
elif DATASET == 'lizard':
    NUM_CLASSES = 6
else:
    raise ValueError('Unknown dataset: {}'.format(DATASET))


batch_size = 64
if DATASET == 'banana':
    train_iter, val_iter = load_data_bananas('../data/banana-detection', batch_size)
elif DATASET == 'lizard':
    dataset = LizardDetectionDataset.from_datadir(
        data_dir=Path('/home/alok/cbmi/data/LizardDataset'),
        image_size=np.array([256, 256]),
        image_stride=np.array([256, 256]),
        use_cache=True,
        show_progress=True,
        force_one_class=True,
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
net = SSDModel(num_classes=NUM_CLASSES, backbone_arch='tiny')
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)


num_epochs = 10
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
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio=3.0)
            l.backward()
            trainer.step()
            metric[0] += cls_eval(cls_preds, cls_labels)
            metric[1] += cls_labels.numel()
            metric[2] += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            metric[3] += bbox_labels.numel()
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    # noinspection PyTypeChecker
    print(f'{len(train_iter.dataset)} examples on {str(device)}')
    if MODEL_LOAD_PATH:
        torch.save(net.state_dict(), MODEL_LOAD_PATH)


# Prediction

mean_average_precision = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

do_display = False

net.eval()
for batch in val_iter:
    images = batch['image']
    ground_truth_boxes = batch['boxes']
    debug(images.shape)

    anchors, cls_preds, bbox_preds = net(images)
    debug(anchors.shape)
    predict_clock = Clock()
    batch_output = predict(anchors, cls_preds, bbox_preds, confidence_threshold=0.7, num_pred_limit=300)
    predict_clock.sap('predict')

    update_mean_average_precision(mean_average_precision, ground_truth_boxes, batch_output)

    if do_display:
        for image, ground_truth_box, output in zip(images, ground_truth_boxes, batch_output):
            draw_image = (image * 255.0).squeeze(0).permute(1, 2, 0).long()

            def display(img, out, boxes):
                for row in out:
                    bbox = row[2:6].unsqueeze(0)
                    draw_boxes(img, bbox, color=(255, 0, 0), box_format='ltrb')
                if DISPLAY_GROUND_TRUTH:
                    for box in boxes:
                        if box[0] < 0:
                            continue
                        bbox = box[1:5].unsqueeze(0)
                        draw_boxes(img, bbox, color=(0, 255, 0), box_format='ltrb')
                plt.imshow(img)
                plt.draw()
                k = plt.waitforbuttonpress()
                plt.close()
                return k

            key = display(draw_image, output, ground_truth_box)
            if not key:
                do_display = False
                break
    break

mean_average_precision_clock = Clock()
mean_ap = mean_average_precision.compute()
mean_average_precision_clock.sap('map.compute()')
pprint(mean_ap)
