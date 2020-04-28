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
    def __init__(self, hidden_dim=1024, image_channels=3, mode='single-image'):
        super().__init__()

        assert mode in MODES

        self.mode = mode
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
        # Output size --> 10x400x400 (Upsampled to 10x800x800)
        # TODO

    def single_image_forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_translation_layer(x)
        x = self.single_image_reconstructor(x)
        return x

    def forward(self, x):
        if self.mode == 'single-image':
            return self.single_image_forward(x)

    def save(self, epoch_num=None, save_dir='resnet_weights'):
        if epoch_num is None:
            epoch_num = 'latest'
        else:
            epoch_num = str(epoch_num) + '-epochs'
        full = os.path.join(save_dir, 'resnet-' + epoch_num + '.torch')
        backbone = os.path.join(save_dir, 'backbone-' + epoch_num + '.torch')
        torch.save(self.state_dict(), full)
        torch.save(self.backbone.state_dict(), backbone)
