import torch
import torch.utils.data
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
    def forward(self, x, size=1024):
        return x.view(x.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super().__init__()
        
        self.device = torch.device('cuda')

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(3, stride=3),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(3, stride=3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(3, stride=3),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(3, stride=3),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # self.decoder = nn.Sequential(
        #     UnFlatten(),
        #     Interpolate(scale_factor=(3, 3), mode='bilinear'),
        #     nn.ConvTranspose2d(h_dim, 128, kernel_size=7, stride=1, padding=0),
        #     nn.ReLU(),
        #     Interpolate(scale_factor=(3, 3), mode='bilinear'),
        #     nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=1),
        #     nn.ReLU(),
        #     Interpolate(scale_factor=(3, 3), mode='bilinear'),
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=2),
        #     nn.ReLU(),
        #     Interpolate(scale_factor=(3, 3), mode='bilinear'),
        #     nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid(),
        # )

        # self.decoder = nn.Sequential(
        #     UnFlatten(),
        #     nn.ConvTranspose2d(h_dim, 128, kernel_size=(5, 5), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1), output_padding=(1, 1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(4, 4), padding=(0, 1), output_padding=(0, 1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, image_channels, kernel_size=(5, 5), stride=(4, 5), padding=(1, 7)),
        #     nn.Sigmoid(),
        # )

        self.decoder = nn.Sequential(
            UnFlatten(),
            Interpolate(scale_factor=(4, 4), mode='bilinear'),
            nn.Conv2d(h_dim, 128, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(32, image_channels, kernel_size=(3, 5), stride=1, padding=(1, 0)),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD
