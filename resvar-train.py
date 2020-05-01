#!/usr/bin/env python
# coding: utf-8

from time import perf_counter

import torch

from data import get_unlabeled_set, get_labeled_set, set_seeds
from model.resnet import Prototype

#
# Parameters
#

# The Resnet prototype will use variational
# representations
variational = True
output_path = '/scratch/css459/resvar_weights/'

unlabeled_batch_size = 32
labeled_batch_size = 16
hidden_size = 1024

unlabeled_epochs = 10
labeled_epochs = 10

# Loads the Unlabeled-trained model from disk
skip_unlabeled_training = True
resume_unlabeled = True

#
# Setup
#

set_seeds()
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    unlabeled_batch_size *= torch.cuda.device_count()
    labeled_batch_size *= torch.cuda.device_count()

#
# Data
#

_, unlabeled_trainloader = get_unlabeled_set(batch_size=unlabeled_batch_size)
(_, labeled_trainloader), (_, labeled_testloader) = get_labeled_set(batch_size=labeled_batch_size,
                                                                    validation=0.2)

#
# Model
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Prototype(device, hidden_dim=hidden_size, variational=variational)

if skip_unlabeled_training or resume_unlabeled:
    print('==> Loading Saved Unlabeled Weights')
    model.load_state_dict(torch.load('./resvar_weights/unlabeled-resnet-latest.torch'))

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    assert unlabeled_batch_size >= torch.cuda.device_count()
    print('==> Using Data Parallel Devices:', torch.cuda.device_count())

model = model.to(device)

print('==> Device:', device)
print('==> Batch Size:', unlabeled_batch_size)
print('==> Unlabeled Epochs:', unlabeled_epochs)
print('==> Labeled Epochs:', labeled_epochs)
print('==> Hidden Encoding Size per Image:', hidden_size)
print('==> Model Loaded. Begin Training')

#
# Unlabeled Pre-training
#

criterion = model.module.loss_function
optimizer = torch.optim.Adam(model.parameters())

model.train()
if not skip_unlabeled_training:

    i = 0
    start_time = perf_counter()
    for epoch in range(unlabeled_epochs):
        loss = 0.0

        max_batches = len(unlabeled_trainloader)
        for idx, (images, camera_index) in enumerate(unlabeled_trainloader):
            optimizer.zero_grad()

            images = images.to(device)
            reconstructions, mu, logvar = model(images, mode='single-image')
            loss, bce, kld = criterion(reconstructions, images,
                                       mode='single-image',
                                       mu=mu,
                                       logvar=logvar,
                                       kld_schedule=0.0015,
                                       i=i)
            loss.backward()
            optimizer.step()

            i += 1

            # Training Wheels
            # print('loss', loss.item())
            # break

            if idx % 1000 == 0:
                print('[', epoch, '|', idx, '/', max_batches, ']', 'loss:', loss.item(),
                      'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))

        Prototype.save(model, file_prefix='unlabeled-', save_dir=output_path)

    print('Unlabeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))

#
# Labeled Training
#

# FREEZE BACKBONE: No more training to the encoder
model.module.freeze_backbone()
print('==> BACKBONE FROZEN: Beginning Labeled Training')

i = 0
start_time = perf_counter()
for epoch in range(labeled_epochs):
    loss = 0.0

    max_batches = len(labeled_trainloader)
    for idx, (images, _, road_map) in enumerate(labeled_trainloader):
        optimizer.zero_grad()

        images = torch.stack(images)
        images = images.to(device)

        # Rasterize bounding box images for reconstruction
        # targets = make_bounding_box_images(targets)
        # targets = targets.to(device)

        road_map = torch.stack(road_map).to(device)

        # print('input shape:', images.shape)
        # print('targt shape:', targets.shape)

        reconstructions, mu, logvar = model(images, mode='road-map')

        # print('outpt shape:', reconstructions.shape)

        loss, bce, kld = criterion(reconstructions, road_map,
                                   mode='road-map',
                                   mu=mu,
                                   logvar=logvar,
                                   kld_schedule=0.05,
                                   i=i)
        loss.backward()
        optimizer.step()

        i += 1

        # Training Wheels
        # print('loss', loss.item())
        # break

        if idx % 10 == 0:
            print('[', epoch, '|', idx, '/', max_batches, ']', 'loss:', loss.item(),
                  'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))

    Prototype.save(model, file_prefix='labeled-roadmap-', save_dir=output_path)

print('Labeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))

