#!/usr/bin/env python
# coding: utf-8

from time import perf_counter
import torch
from data import get_unlabeled_set, get_labeled_set, set_seeds, make_bounding_box_images

from model.resnet import Prototype

#
# Parameters
#

batch_size = 4
hidden_size = 1024

unlabeled_epochs = 3
labeled_epochs = 10

#
# Setup
#

set_seeds()
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    batch_size *= torch.cuda.device_count()

#
# Data
#

_, unlabeled_trainloader = get_unlabeled_set(batch_size=batch_size)
(_, labeled_trainloader), (_, labeled_testloader) = get_labeled_set(batch_size=batch_size, validation=0.2)

#
# Model
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Prototype(hidden_dim=hidden_size)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    assert batch_size >= torch.cuda.device_count()
    print('==> Using Data Parallel Devices:', torch.cuda.device_count())

model = model.to(device)

print('==> Device:', device)
print('==> Batch Size:', batch_size)
print('==> Unlabeled Epochs:', unlabeled_epochs)
print('==> Labeled Epochs:', labeled_epochs)
print('==> Hidden Encoding Size per Image:', hidden_size)
print('==> Model Loaded. Begin Training')

#
# Unlabeled Pre-training
#

criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())

model.train()

start_time = perf_counter()
for epoch in range(unlabeled_epochs):
    loss = 0.0

    max_batches = len(unlabeled_trainloader)
    for idx, (images, camera_index) in enumerate(unlabeled_trainloader):
        optimizer.zero_grad()

        images = images.to(device)
        reconstructions = model(images, mode='single-image')
        loss = criterion(reconstructions, images)

        loss.backward()
        optimizer.step()

        # Training Wheels
        # print('loss', loss.item())
        # break

        if idx % 1000 == 0:
            print('[', epoch, '|', idx, '/', max_batches, ']', 'loss:', loss.item(),
                  'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))

    Prototype.save(model, file_prefix='unlabeled-')

print('Unlabeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))

#
# Labeled Training
#

model.train()

start_time = perf_counter()
for epoch in range(labeled_epochs):
    loss = 0.0

    max_batches = len(labeled_trainloader)
    for idx, (images, targets, _) in enumerate(labeled_trainloader):
        optimizer.zero_grad()

        # Restack images to be (6-directions, batch size, channels, H, W)
        images = torch.stack(images).permute(1, 0, 2, 3, 4)
        images = images.to(device)

        # Rasterize bounding box images for reconstruction
        targets = make_bounding_box_images(targets)
        targets = targets.to(device)

        # print('input shape:', images.shape)
        # print('targt shape:', targets.shape)

        reconstructions = model(images, mode='object-map')

        # print('outpt shape:', reconstructions.shape)

        loss = criterion(reconstructions, targets)

        loss.backward()
        optimizer.step()

        # Training Wheels
        # print('loss', loss.item())
        # break

        if idx % 1000 == 0:
            print('[', epoch, '|', idx, '/', max_batches, ']', 'loss:', loss.item(),
                  'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))

    Prototype.save(model, file_prefix='labeled-')

print('Labeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))

