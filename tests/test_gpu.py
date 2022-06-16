import torch


torch.set_printoptions(2)


def main():
    a = torch.tensor([1, 2, 3], device='cuda:0')
    print(a.device)
    print(type(a.device))


if __name__ == '__main__':
    main()
