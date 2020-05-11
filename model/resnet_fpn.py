import torch
from torch import nn
from torch.nn.functional import interpolate
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class BEVProjection(nn.Module):
    def __init__(self, input_channels=1, output_dim=400, output_channels=1,
                 camera_focal_length=0.0, camera_horz_offset=0.0):
        super().__init__()


class DenseTransformerLayer(nn.Module):
    """
    Adapted Methods from:
        Predicting Semantic Map Representations from Images
        using Pyramid Occupancy Networks
            https://arxiv.org/pdf/2003.13402.pdf
    """

    def __init__(self, input_size=64, input_channels=256, bottleneck_channels=2,
                 bottleneck_size=1, output_depth=13):
        super().__init__()
        self.input_size = input_size
        self.activation = nn.LeakyReLU(inplace=True)

        self.vertical_collapse = nn.Linear(input_size, bottleneck_size)
        self.channel_collapse = nn.Linear(input_channels, bottleneck_channels)

        self.depth_expansion = nn.Linear(bottleneck_size, output_depth)

    def forward(self, x):
        # Vertical, Channel collapse
        x = self.vertical_collapse(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.activation(x)
        x = self.channel_collapse(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x = self.activation(x)
        x = self.depth_expansion(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        # ... Polar BEV Projection here?

        return x


class PyramidPoolingLayer(nn.Module):
    def __init__(self, output_size=(64, 64)):  # pyramid_size=5, channels=256
        super().__init__()

        # Averages over up-sampled feature maps stacked
        # along their channel dimension
        # self.feature_avg_conv = nn.Conv2d(pyramid_size * channels, channels, 1)

        self.max_pool = nn.AdaptiveMaxPool2d(output_size)
        self.activation = nn.Tanh()

        # Each level of the pyramid will have its own
        # Dense Transformer Layer
        self.transformers = [DenseTransformerLayer(input_size=64),
                             DenseTransformerLayer(input_size=32),
                             DenseTransformerLayer(input_size=16),
                             DenseTransformerLayer(input_size=8),
                             DenseTransformerLayer(input_size=4)]

    def forward(self, x):
        # Run all pyramids maps to discrete Dense
        # transformers, and then re-sample along their
        # width by their respective down-sampling relative
        # to the input image.

        # down_sampling_factor = 2**(layer_num + 2)
        scale = 1
        maps = []
        for i, key in enumerate(x):
            m = x[key]
            m = self.transformers[i](m)
            m = interpolate(m,
                            scale_factor=(1, scale),
                            mode='bilinear',
                            align_corners=False)
            maps.append(m)
            scale *= 2

        # Concat maps along depth dim
        maps = torch.cat(maps, dim=2)

        return self.activation(self.max_pool(maps))


class ResnetFPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet_id = 'resnet50'
        self.resnet_output_channels = 256
        self.resnet_pyramid_size = 5
        self.resnet_output_size = 64

        # Forward projection of input image
        # into BEV
        self.project = nn.Sequential(

            # Make feature maps pyramid
            resnet_fpn_backbone(self.resnet_id,
                                pretrained=False),

            # Transform the feature map coordinates to BEV
            PyramidPoolingLayer()
        )

    def forward(self, x):
        """
        The input X assumes six images given in the order:
            Front left, front, front right,
            back left, back,  back right
        The resulting projected feature maps will be
        concatenated in 2D in the above configuration.

        :param x:   Input tensor [batch, 6, c, h, w]
        :return:    Feature map [batch, c_out, h_out, w_out]
        """
        front = []  # Front-facing 3 feature maps
        back = []  # Back-facing 3 feature maps

        x = x.permute(1, 0, 2, 3, 4)
        for i, img in enumerate(x):
            if i < 3:
                front.append(self.project(img))
            else:
                back.append(self.project(img))

        # Concat fronts and backs
        front = torch.cat(front, dim=3)
        back = torch.cat(back, dim=3)

        # Concat front and back
        return torch.cat([front, back], dim=2)

    def infer(self, x):
        with torch.no_grad:
            self.eval()
            return torch.sigmoid(self.forward(x))


class MapReconstructor(nn.Module):
    def __init__(self, input_channels, output_channels=1, output_size=(400, 400), scale=2):
        super().__init__()

        self.activation = nn.Tanh()

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, stride=2),
            nn.AdaptiveMaxPool2d((200, 200)),
            self.activation,
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=2),
            self.activation,
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3),
            self.activation,
            nn.AdaptiveMaxPool2d(output_size),
            nn.Upsample(scale_factor=(scale, scale), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.decode(x)

    def infer(self, x):
        with torch.no_grad:
            self.eval()
            return torch.sigmoid(self.forward(x))
