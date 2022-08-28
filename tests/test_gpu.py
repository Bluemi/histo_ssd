import torch

from utils.funcs import debug


def main():
    a = torch.tensor([[-1, 0, 0], [0, 1, 1], [-1, 2, 2], [1, 3, 3]])
    debug(a)
    labels = a[:, 0]
    debug(labels)
    indices = torch.where(labels >= 0)[0]
    debug(indices)
    debug(a[indices])


if __name__ == '__main__':
    main()
