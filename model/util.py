from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        y = self.interp(x,
                        scale_factor=self.scale_factor,
                        mode=self.mode,
                        align_corners=False)
        return y


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        return x.view(x.size(0), self.input_size, 1, 1)


def remove_backbone_head(model):
    return nn.Sequential(*(list(model.children())[:-1])), model.fc.in_features
