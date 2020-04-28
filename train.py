import random

import numpy as np
import torch
import torch.nn as nn
import torchvision

from helpers.data_helper import UnlabeledDataset, LabeledDataset
from helpers.helper import collate_fn
from model.vae import VAE, vae_loss_function

#
# Setup
#

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

batch_multiplier = 4

#
# Data
#

image_folder = '../data'
annotation_csv = '../data/annotation.csv'

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

transform = torchvision.transforms.ToTensor()

unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                      scene_index=labeled_scene_index,
                                      first_dim='image',  # 'sample'
                                      transform=transform)

unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset,
                                                    batch_size=4 * batch_multiplier,
                                                    shuffle=True,
                                                    num_workers=2)

# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=transform,
                                  extra_info=True
                                  )
labeled_trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                                  batch_size=2,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  collate_fn=collate_fn)

#
# Model
#

device = torch.device('cuda')

epochs = 50
hidden_size = 1024
latent_size = 512

model = VAE(h_dim=hidden_size, z_dim=latent_size)
model = nn.DataParallel(model)
model = model.to(device)

criterion = vae_loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#
# Training
#

model.train()
for epoch in range(epochs):
    loss, bce, kld = 0, 0, 0

    max_batches = len(unlabeled_trainloader)
    for idx, (images, camera_index) in enumerate(unlabeled_trainloader):
        images = images.to(device)
        recon_images, mu, logvar = model(images)

        loss, bce, kld = criterion(recon_images, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1000 == 0:
            print('[', epoch, '|', idx, '/', max_batches, ']',
                  'loss:', loss.item(), 'bce:', bce.item(), 'kld:', kld.item())

    torch.save(model.state_dict(), 'vae-epoch-latest.torch')
