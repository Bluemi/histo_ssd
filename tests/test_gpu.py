import torch


def main():
    a = torch.arange(0, 4).reshape((2, 2))
    b = torch.arange(0, 4).reshape((2, 2)) + 4
    print(f'matmul\n{a}\n{b}\n)')
    print(torch.matmul(a, b))


if __name__ == '__main__':
    main()
