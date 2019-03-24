import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def Net(in_ch, out_ch):
    net = nn.Sequential(
        nn.Conv1d(in_channels=in_ch, out_channels=32, kernel_size=7),
        nn.LeakyReLU(),
        nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3),
        nn.LeakyReLU(),
        nn.MaxPool1d(kernel_size=2),
        Flatten(),
        nn.Linear(in_features=2848, out_features=1000),
        nn.LeakyReLU(),
        nn.Linear(in_features=1000, out_features=out_ch),
    )
    return net
