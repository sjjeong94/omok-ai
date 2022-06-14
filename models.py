import torch
import torch.nn as nn


def ConvBNReLU(in_ch, out_ch, k_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k_size, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, ch, k_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, k_size, stride, padding),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, k_size, stride, padding),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = x + res
        x = self.act(x)
        return x


class Tower(nn.Module):
    def __init__(self, in_ch=1, channels=64, num_blocks=5, k_size=3):
        super().__init__()
        padding = k_size // 2
        self.conv1 = ConvBNReLU(in_ch, channels, k_size, 1, padding)
        blocks = [ResBlock(channels, k_size, 1, padding)
                  for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.conv2 = ConvBNReLU(channels, 2, 1, 1, 0)
        self.flat = nn.Flatten()
        self.lin = nn.Linear(2 * 15 * 15, 15 * 15)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.lin(x)
        return x


if __name__ == '__main__':
    net = Tower()
    print(net)
    x = torch.randn(1, 1, 15, 15)
    y = net(x)
    print(x.shape, y.shape)
