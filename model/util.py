from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if self.mode == 'nearest':
            y = self.interp(x,
                            scale_factor=self.scale_factor,
                            mode=self.mode)
        else:
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


class InterpolatingDecoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        # Bounding-box-image reconstruction
        # Output size --> 3x400x400 (Interpolated to 10x800x800)
        #
        # INPUT: (batch size, 6144, 1, 1)
        # Let dim = 6144:
        #   Downscale dim by factor of 4 --> 4 Times
        #   Then, downscale dim by factor of 2 --> 1 Time
        #   Them downscale two dimensions for a final dim of 3
        #
        # OUTPUT: (batch size, 3, 400, 400)
        #          --> Call UpSample(2,2)
        #          --> (batch size, 3, 800, 800)
        self.input_dim = 6144
        self.map_reconstructor = nn.Sequential(
            UnFlatten(input_size=6144),

            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(hidden_dim * 6, 1536, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(),

            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(1536, 384, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(384, 96, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(96, 24, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            Interpolate(scale_factor=(2, 2), mode='bilinear'),
            nn.Conv2d(24, 12, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

    def forward(self, x):
        return self.map_reconstructor(x)


def remove_backbone_head(model):
    return nn.Sequential(*(list(model.children())[:-1])), model.fc.in_features
