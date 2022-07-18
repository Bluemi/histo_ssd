import torch

from models import TinySSD
from utils.funcs import debug


def main():
    net = TinySSD(num_classes=1)
    x = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(x)
    debug(anchors.shape)
    debug(cls_preds.shape)
    debug(bbox_preds.shape)


if __name__ == '__main__':
    main()
