import torch
from d2l.torch import d2l
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from datasets.banana_dataset import load_data_bananas
from models import SSDModel


def main():
    net = SSDModel(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)

    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)

    batch_size = 32
    train_iter, _ = load_data_bananas('../data/banana-detection', batch_size)

    device, net = 'cpu', SSDModel(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        cls = cls_loss(cls_preds.reshape(-1, num_classes),
                       cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(bbox_preds * bbox_masks,
                         bbox_labels * bbox_masks).mean(dim=1)
        return cls + bbox

    def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor):
        # Because the class prediction results are on the final dimension,
        # `argmax` needs to specify this dimension
        return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

    # --------------------

    num_epochs, timer = 20, d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
    net = net.to('cpu')
    cls_err = None
    bbox_mae = None
    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        print(f'--- epoch {epoch} ---')
        metric = d2l.Accumulator(4)
        net.train()
        for features, target in tqdm(train_iter):
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            # Calculate the loss function using the predicted and labeled values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(
                cls_eval(cls_preds, cls_labels),
                cls_labels.numel(),
                bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                bbox_labels.numel()
            )
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')


if __name__ == '__main__':
    main()
