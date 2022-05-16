import torch


def main():
    print('cuda_available:', torch.cuda.is_available())
    print('device name:', torch.cuda.get_device_name(0))


if __name__ == '__main__':
    main()
