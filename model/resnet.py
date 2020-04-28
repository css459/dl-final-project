import os

import torch
from torch import nn
from torchvision.models.resnet import resnet18

from model.util import Interpolate, UnFlatten, remove_backbone_head

MODES = ['single-image',
         'object-map',
         'road-map'
         'jigsaw-pretext']


class Prototype(nn.Module):
    def __init__(self, hidden_dim=1024, image_channels=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_image_channels = image_channels

        #
        # Backbone
        #

        self.backbone, in_features = remove_backbone_head(resnet18(pretrained=False,
                                                                   progress=True))

        # Translates the expected size of the backbone output to our
        # hidden size for later networks
        self.fc_translation_layer = nn.Linear(in_features, hidden_dim)

        #
        # Different "Heads" for which to fit onto the backbone
        #

        # Single-image reconstruction
        # Output size --> Input size
        self.single_image_reconstructor_input_dim = hidden_dim
        self.single_image_reconstructor = nn.Sequential(
            UnFlatten(input_size=hidden_dim),

            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            Interpolate(scale_factor=(4, 5), mode='bilinear'),
            nn.Conv2d(32, image_channels, kernel_size=(3, 5), stride=1, padding=(1, 0))
        )

        # Bounding-box-image reconstruction
        # Output size --> 10x400x400 (Interpolated to 10x800x800)
        #
        # INPUT: (batch size, 6144, 1, 1)
        # Let dim = 6144:
        #   Downscale dim by factor of 4 --> 4 Times
        #   Then, downscale dim by factor of 2 --> 1 Time
        #   Them downscale two dimensions for a final dim of 10
        #
        # OUTPUT: (batch size, 10, 400, 400)
        #          --> Call UpSample(2,2)
        #          --> (batch size, 10, 800, 800)
        self.object_map_reconstructor = nn.Sequential(
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
            nn.ReLU(),

            nn.Conv2d(12, 10, kernel_size=7, stride=1, padding=1),
            Interpolate(scale_factor=(2, 2), mode='bilinear'),
        )

    def backbone_encode(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_translation_layer(x)
        return x

    def single_image_forward(self, x):
        x = self.backbone_encode(x)
        x = self.single_image_reconstructor(x)
        return x

    def object_map_forward(self, x):
        # Encode all 6 images along the
        # second dimension
        acc = self.backbone_encode(x[0])
        for i in x[1:]:
            enc = self.backbone_encode(i)
            acc = torch.cat((acc, enc), 1)

        acc = self.object_map_reconstructor(acc)
        return acc

    def forward(self, x, mode='single-image'):
        assert mode in MODES

        if mode == 'single-image':
            return self.single_image_forward(x)

        if mode == 'object-map':
            return self.object_map_forward(x)

    @staticmethod
    def save(model, using_dataparallel=True, epoch_num=None, file_prefix='', save_dir='./resnet_weights'):
        if epoch_num is None:
            epoch_num = 'latest'
        else:
            epoch_num = str(epoch_num) + '-epochs'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        full = os.path.join(save_dir, file_prefix + 'resnet-' + epoch_num + '.torch')
        backbone = os.path.join(save_dir, file_prefix + 'backbone-' + epoch_num + '.torch')
        if using_dataparallel:
            torch.save(model.module.state_dict(), full)
            torch.save(model.module.backbone.state_dict(), backbone)
        else:
            torch.save(model.state_dict(), full)
            torch.save(model.backbone.state_dict(), backbone)
