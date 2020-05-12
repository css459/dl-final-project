#!/usr/bin/env python
# coding: utf-8

import os
from time import perf_counter

import torch

from data import get_unlabeled_set, get_labeled_set, make_bounding_box_images, convert_bounding_box_targets
from model.resnet import Prototype
from model.segmentation import SegmentationNetwork

#
# Parameters
#

# The Resnet prototype will use variational
# representations
variational = True
output_path = '/scratch/css459/resvar_weights/'

unlabeled_batch_size = 32
labeled_batch_size = 2

unlabeled_epochs = 10
labeled_epochs = 30

# Loads the Unlabeled-trained model from disk
skip_unlabeled_training = True
load_unlabeled = True
load_labeled = True

#
# Setup
#

# set_seeds()
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
model = Prototype(device, hidden_dim=1024, variational=variational)
seg_model = SegmentationNetwork()

if load_unlabeled and not load_labeled:
    print('==> Loading Saved Unlabeled Weights (Backbone)')
    file_path = os.path.join(output_path, 'unlabeled-resvar-backbone-latest.torch')
    model.load_backbone(file_path)

elif load_labeled:
    print('==> Loading Saved Labeled Weights (Backbone)')
    file_path = os.path.join(output_path, 'labeled-resvar-latest.torch')
    # model.load_backbone(file_path)
    model.load_state_dict(torch.load(file_path))

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    assert unlabeled_batch_size >= torch.cuda.device_count()
    print('==> Using Data Parallel Devices:', torch.cuda.device_count())

model = model.to(device)
seg_model.to(device)

print('==> Device:', device)
print('==> Batch Size:', unlabeled_batch_size)
print('==> Unlabeled Epochs:', unlabeled_epochs)
print('==> Labeled Epochs:', labeled_epochs)
print('==> Model Loaded. Begin Training')

#
# Unlabeled Pre-training
#

criterion = model.module.loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

model.train()
seg_model.train()
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
# model.module.freeze_backbone()
# print('==> BACKBONE FROZEN: Beginning Labeled Training')

i = 0
start_time = perf_counter()
for epoch in range(labeled_epochs):
    loss = 0.0

    max_batches = len(labeled_trainloader)
    for idx, (images, targets, road_map) in enumerate(labeled_trainloader):
        optimizer.zero_grad()

        # Format inputs
        images = torch.stack(images)
        images = images.to(device)

        # Rasterize bounding box images for reconstruction
        targets_img = make_bounding_box_images(targets)
        targets_img = targets_img.to(device)

        # Make the targets for the Segmentation Network
        targets_seg = convert_bounding_box_targets(targets, device)

        # Format road map
        road_map = torch.stack(road_map).float()
        road_map = road_map.view(-1, 1, 800, 800)
        road_map = road_map.to(device)

        # Forward
        obj_recon, road_recon, mu, logvar = model(images, mode='object-road-maps')

        # Reconstruction losses
        road_loss, _, _ = criterion(road_recon, road_map,
                                    mode='road-map',
                                    mu=mu,
                                    logvar=logvar,
                                    loss_reduction='mean',
                                    kld_schedule=0.05,
                                    i=i)

        obj_loss, _, obj_kld = criterion(obj_recon, targets_img,
                                         mode='road-map',
                                         mu=mu,
                                         logvar=logvar,
                                         loss_reduction='mean',
                                         kld_schedule=0.05,
                                         i=i)

        # Backward through the road map head separately
        road_loss.backward(retain_graph=True)
        loss = obj_loss

        # Start training the Segmentation Network after 1 Epoch
        if epoch == 2:
            model.module.freeze_backbone()
            print('==> BACKBONE FROZEN')
        if epoch > 1:
            seg_losses = seg_model(obj_recon, targets_seg)
            seg_losses = seg_losses['loss_box_reg'] + seg_losses['loss_rpn_box_reg']
            seg_losses.backward(retain_graph=True)
            # loss += seg_losses

        loss.backward()
        optimizer.step()

        i += 1

        # Training Wheels
        # print('loss', loss.item())
        # break

        if idx % 10 == 0:
            if epoch > 1:
                print('[', epoch, '|', idx, '/', max_batches, ']', 'loss:', loss.item(), 'seg loss', seg_losses.item(),
                      'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))
            else:
                print('[', epoch, '|', idx, '/', max_batches, ']', 'loss:', loss.item(),
                      'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))

    Prototype.save(model, file_prefix='labeled-', save_dir=output_path)
    torch.save(seg_model.state_dict(), output_path + 'segmentation-network-latest.torch')

print('Labeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))
