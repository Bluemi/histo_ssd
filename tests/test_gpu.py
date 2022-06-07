import torch


torch.set_printoptions(2)
OFFSET1 = 0.01
OFFSET2 = 0.02
OFFSET3 = 0.03
OFFSET4 = 0.04
NUM_POINTS = 3


def main():
    # a = torch.tensor([3, 5])
    a = torch.tensor([[3, 5], [4, 6], [5, 7]])
    a = a.reshape((-1, 1, 1, 2))
    b = torch.tensor(
        [[[-OFFSET1, -OFFSET1], [OFFSET1, OFFSET1]],
         [[-OFFSET2, -OFFSET2], [OFFSET2, OFFSET2]],
         [[-OFFSET3, -OFFSET3], [OFFSET3, OFFSET3]],
         [[-OFFSET4, -OFFSET4], [OFFSET4, OFFSET4]]]
    )
    b = b.reshape((1, 4, 2, 2))
    # b = torch.stack([b]*3)

    print('a shape: {}'.format(a.size()))
    print('b shape: {}'.format(b.size()))
    print('\na:')
    print(a)
    print('b:')
    print(b)

    result = torch.add(a, b)
    print('\nresult.shape: {}'.format(result.size()))
    print('result:')
    print(result)


if __name__ == '__main__':
    main()
