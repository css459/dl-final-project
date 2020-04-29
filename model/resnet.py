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

        resnet = resnet18(pretrained=False, progress=True)
        self.backbone, in_features = remove_backbone_head(resnet)

        # Translates the expected size of the backbone output to our
        # hidden size for later networks
        self.fc_translation_layer = nn.Linear(in_features, hidden_dim)

        #
        # Variational Layers
        #
        # These layers are used when the hidden encoding is to be variational
        #

        self.device = None
        self.is_variational = False

        z_dim = 512
        self.fc1_var_encode = nn.Linear(hidden_dim, z_dim)
        self.fc2_var_encode = nn.Linear(hidden_dim, z_dim)
        self.fc3_var_decode = nn.Linear(z_dim, hidden_dim)

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
        self.single_image_reconstructor_input_dim = 6144
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

    #
    # Variational Functions
    #

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu = self.fc1_var_encode(h)
        logvar = self.fc2_var_encode(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    #
    # Encoders
    #

    def backbone_encode(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_translation_layer(x)
        return x

    def backbone_variational_encode(self, x):
        x = self.backbone_encode(x)
        z, mu, logvar = self.bottleneck(x)
        return z, mu, logvar

    def variational_decode(self, decode_fn, z):
        z = self.fc3_var_decode(z)
        return decode_fn(z)

    #
    # Forward Functions
    #

    def single_image_forward(self, x):
        x = self.backbone_encode(x)
        x = self.single_image_reconstructor(x)
        return x

    def single_image_variational_forward(self, x):
        z, mu, logvar = self.backbone_variational_encode(x)
        z = self.variational_decode(self.single_image_reconstructor, z)
        z = torch.sigmoid(z)

        return z, mu, logvar

    def object_map_forward(self, x):
        # Re-stack images to be
        # (6-directions, batch size, channels, H, W)
        x = x.permute(1, 0, 2, 3, 4)

        # Encode all 6 images along the
        # second dimension
        acc = self.backbone_encode(x[0])
        for i in x[1:]:
            enc = self.backbone_encode(i)
            acc = torch.cat((acc, enc), 1)

        if self.is_variational:
            z, mu, logvar = self.backbone_variational_encode(x)

        acc = self.object_map_reconstructor(acc)
        return acc

    def object_map_variational_forward(self, x):
        # Re-stack images to be
        # (6-directions, batch size, channels, H, W)
        x = x.permute(1, 0, 2, 3, 4)

        # print('x', x.shape)

        # Encode all 6 images along the
        # second dimension
        z_acc, mu_acc, logvar_acc = self.backbone_variational_encode(x[0])

        # print('z acc', z_acc.shape)
        # print('mu acc', mu_acc.shape)
        # print('logvar acc', logvar_acc.shape)

        for i in x[1:]:
            z, mu, logvar = self.backbone_variational_encode(i)
            z_acc = torch.cat((z_acc, z), 1)
            mu_acc = torch.cat((mu_acc, mu), 1)
            logvar_acc = torch.cat((logvar_acc, logvar), 1)

        # Combination layer for latent concatenation
        # NOTE: Is this correct to do this?
        fc_latent_translation = nn.Linear(z_acc.size(1),
                                          self.hidden_dim * x.size(0)).to(self.device)
        z_acc = fc_latent_translation(z_acc)

        # print('z acc', z_acc.shape)
        # print('mu acc', mu_acc.shape)
        # print('logvar acc', logvar_acc.shape)

        # Call the reconstructor directly, we've already done the FC as above
        z_acc = self.object_map_reconstructor(z_acc)
        z_acc = torch.sigmoid(z_acc)

        # print('z acc dec', z_acc.shape)

        return z_acc, mu_acc, logvar_acc

    def forward(self, x, mode='single-image'):
        assert mode in MODES

        if mode == 'single-image':
            if self.is_variational:
                return self.single_image_variational_forward(x)
            else:
                return self.single_image_forward(x)

        if mode == 'object-map':
            if self.is_variational:
                return self.object_map_variational_forward(x)
            else:
                return self.object_map_forward(x)

    #
    # Utility Functions
    #

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

    # Reconstruction + KL divergence losses summed over all elements and batch
    @staticmethod
    def var_loss_function(recon_x, x, mu, logvar):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD
