import os

import torch
from torch import nn
from torchvision.models.resnet import resnet18

from model.util import Interpolate, UnFlatten, InterpolatingDecoder, remove_backbone_head

MODES = ['single-image',
         'object-map',
         'road-map',
         'object-road-maps',
         'jigsaw-pretext']


class Prototype(nn.Module):
    def __init__(self, device, hidden_dim=1024, image_channels=3,
                 output_channels=3, image_stack_count=6, variational=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_image_channels = image_channels
        self.output_channels = output_channels
        self.image_stack_count = image_stack_count

        self.device = device
        self.is_variational = variational

        # Class Weights for bounding-box image generator
        # [Background, Car, Other Object]
        self.object_weights = [0.05, 0.9, 0.05]

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

        z_dim = 512
        self.fc1_var_encode = nn.Linear(hidden_dim, z_dim)
        self.fc2_var_encode = nn.Linear(hidden_dim, z_dim)
        self.fc3_var_decode = nn.Linear(z_dim, hidden_dim)

        self.fc_latent_translation = nn.Linear(z_dim * self.image_stack_count,
                                               self.hidden_dim * self.image_stack_count)

        #
        # Different "Heads" for which to fit onto the backbone
        #

        # Single-image reconstruction
        # Output size --> Input size
        self.map_reconstructor_input_dim = hidden_dim
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

        self.map_reconstructor_input_dim = 6144

        # Reconstructs an image containing the heat-maps
        # of potential object bounding-boxes
        self.object_map_reconstructor = nn.Sequential(
            InterpolatingDecoder(hidden_dim=hidden_dim),
            nn.Conv2d(12, 3, kernel_size=7, stride=1, padding=1),
            Interpolate(scale_factor=(2, 2), mode='nearest')
        )

        # Reconstructs the binary road map
        self.road_map_reconstructor = nn.Sequential(
            InterpolatingDecoder(hidden_dim=hidden_dim),
            nn.Conv2d(12, 1, kernel_size=7, stride=1, padding=1),
            Interpolate(scale_factor=(2, 2), mode='nearest')
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

    def encoded_stack(self, x):
        # Re-stack images to be
        # (6-directions, batch size, channels, H, W)
        x = x.permute(1, 0, 2, 3, 4)

        # Encode all 6 images along the
        # second dimension
        acc = self.backbone_encode(x[0])
        for i in x[1:]:
            enc = self.backbone_encode(i)
            acc = torch.cat((acc, enc), 1)

        # acc = self.map_reconstructor(acc)
        return acc

    def encoded_variational_stack(self, x):
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
        z_acc = self.fc_latent_translation(z_acc)

        # print('z acc', z_acc.shape)
        # print('mu acc', mu_acc.shape)
        # print('logvar acc', logvar_acc.shape)

        # Call the reconstructor directly, we've already done the FC as above
        # z_acc = self.map_reconstructor(z_acc)
        # z_acc = torch.sigmoid(z_acc)

        # print('z acc dec', z_acc.shape)

        return z_acc, mu_acc, logvar_acc

    def forward(self, x, mode):
        assert mode in MODES

        if mode == 'single-image':
            if self.is_variational:
                return self.single_image_variational_forward(x)
            else:
                return self.single_image_forward(x)

        if mode == 'object-map':
            if self.is_variational:
                x, mu, logvar = self.encoded_variational_stack(x)
                x = self.object_map_reconstructor(x)
                # x = torch.sigmoid(x)
                return x, mu, logvar
            else:
                return self.object_map_reconstructor(self.encoded_stack(x))

        if mode == 'road-map':
            if self.is_variational:
                x, mu, logvar = self.encoded_variational_stack(x)
                x = self.road_map_reconstructor(x)
                # x = torch.sigmoid(x)
                return x, mu, logvar
            else:
                return self.road_map_reconstructor(self.encoded_stack(x))

        # Combines the two in a single encoder pass to speed up training
        if mode == 'object-road-maps':
            if self.is_variational:
                x, mu, logvar = self.encoded_variational_stack(x)
                x_obj = self.object_map_reconstructor(x)
                x_road = self.road_map_reconstructor(x)
                # x = torch.sigmoid(x)
                return x_obj, x_road, mu, logvar
            else:
                x = self.encoded_stack(x)
                return self.object_map_reconstructor(x), self.road_map_reconstructor(x)

    #
    # Inference
    #

    def infer_road_map(self, x, threshold=0.5):
        self.eval()
        if self.is_variational:
            x, _, _ = self.forward(x, 'road-map')
        else:
            x = self.forward(x, 'road-map')

        return torch.where(torch.sigmoid(x) > threshold,
                           torch.ones(x.shape), torch.zeros(x.shape))

    def infer_object_heat_map(self, x):
        self.eval()
        if self.is_variational:
            x, _, _ = self.forward(x, 'object-map')
        else:
            x = self.forward(x, 'object-map')

        return torch.softmax(x, 1)

    def infer_single_image(self, x):
        self.eval()
        if self.is_variational:
            x, _, _ = self.forward(x, 'single-image')
        else:
            x = self.forward(x, 'single-image')

        return x

    #
    # Utility Functions
    #

    def freeze_backbone(self, unfreeze=False):
        for p in self.backbone.parameters():
            p.requires_grad = unfreeze

    def load_backbone(self, file_path):
        weights = torch.load(file_path, map_location=self.device)
        self.backbone.load_state_dict(weights)

    @staticmethod
    def save(model, using_data_parallel=True, epoch_num=None,
             file_prefix='', save_dir='./resnet_weights'):

        if epoch_num is None:
            epoch_num = 'latest'
        else:
            epoch_num = str(epoch_num) + '-epochs'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        full = os.path.join(save_dir, file_prefix + 'resvar-' + epoch_num + '.torch')
        backbone = os.path.join(save_dir, file_prefix + 'backbone-' + epoch_num + '.torch')

        if using_data_parallel:
            torch.save(model.module.state_dict(), full)
            torch.save(model.module.backbone.state_dict(), backbone)
        else:
            torch.save(model.state_dict(), full)
            torch.save(model.backbone.state_dict(), backbone)

    def loss_function(self, recon_x, x, mode, mu=None, logvar=None,
                      loss_reduction='sum', kld_schedule=0.0002, i=None):
        assert mode in MODES

        if (mu is None or logvar is None) and self.is_variational:
            raise ValueError('missing required parameters mu and logvar '
                             'for variational loss')

        # How the losses are aggregated over the batch
        assert loss_reduction in ['sum', 'mean']

        # Apply MSE loss over image pixels
        if mode == 'single-image':
            loss_fn = nn.MSELoss(reduction=loss_reduction)

        # LogSoftmax over the channels to compute probability of
        # class under pixel for channel
        elif mode == 'object-map':
            w = torch.FloatTensor(self.object_weights).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=w, reduction=loss_reduction)

        # LogSoftmax over the channels to compute probability of
        # binary class under pixel for channel
        elif mode == 'road-map':
            loss_fn = nn.BCEWithLogitsLoss(reduction=loss_reduction)
        else:
            raise ValueError('Unexpected Mode:', mode)

        loss = loss_fn(recon_x, x)

        # BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if self.is_variational:

            # KLD will be scheduled to increase by `kld_schedule` for each `i`
            # up until 1.0
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            if i and kld_schedule:
                kld *= min(kld_schedule * i, 1.0)
            return loss + kld, loss, kld
        else:
            return loss
