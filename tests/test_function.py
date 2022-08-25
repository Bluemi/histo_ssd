import torch
from skimage.io import imread
from models import SSDModel


NUM_SAMPLES1 = 2
torch.set_printoptions(2)


def main():
    model = SSDModel(num_classes=6, backbone_arch='vgg16', min_anchor_size=0.2, max_anchor_size=0.9)
    print(model.sizes)


if __name__ == '__main__':
    main()
