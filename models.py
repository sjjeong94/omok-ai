import torch
import torch.nn as nn


def TestNet():
    return nn.Sequential(
        nn.Conv2d(1, 32, 7, 1, 3),
        nn.ReLU(),
        nn.Conv2d(32, 64, 7, 1, 3),
        nn.ReLU(),
        nn.Conv2d(64, 128, 7, 1, 3),
        nn.ReLU(),
        nn.Conv2d(128, 64, 7, 1, 3),
        nn.ReLU(),
        nn.Conv2d(64, 32, 7, 1, 3),
        nn.ReLU(),
        nn.Conv2d(32, 1, 1),
        nn.Flatten(),
    )


if __name__ == '__main__':
    net = TestNet()
    x = torch.randn(1, 1, 15, 15)
    y = net(x)
    print(x.shape, y.shape)
