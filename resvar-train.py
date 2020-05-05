#!/usr/bin/env python
# coding: utf-8

import os
from time import perf_counter

import torch

from data import get_unlabeled_set, get_labeled_set, set_seeds, \
    make_bounding_box_images, convert_bounding_box_targets
from model.resnet import Prototype
from model.segmentation import SegmentationNetwork

#
# Parameters
#

# The Resnet prototype will use variational
# representations
variational = True
#output_path = '/scratch/css459/resvar_weights/'
output_path = '../'

unlabeled_batch_size = 32
labeled_batch_size = 4
hidden_size = 1024

unlabeled_epochs = 10
labeled_epochs = 10

# Loads the Unlabeled-trained model from disk
skip_unlabeled_training = True
load_unlabeled = True
load_labeled = True

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
seg_model = SegmentationNetwork(model.backbone, 3)

if load_unlabeled and not load_labeled:
    print('==> Loading Saved Unlabeled Weights (Backbone)')
    file_path = os.path.join(output_path, 'unlabeled-backbone-latest.torch')
    model.load_backbone(file_path)
    # model.load_state_dict(torch.load('./resvar_weights/unlabeled-resnet-latest.torch'))

elif load_labeled:
    print('==> Loading Saved Labeled Weights (Backbone)')
    file_path = os.path.join(output_path, 'labeled-roadmap-backbone-latest.torch')
    # model.load_backbone(file_path)
    model.load_state_dict(torch.load('../labeled-roadmap-backbone-latest.torch'))

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
print('==> Hidden Encoding Size per Image:', hidden_size)
print('==> Model Loaded. Begin Training')

#
# Unlabeled Pre-training
#

criterion = model.module.loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

        images = torch.stack(images)
        images = images.to(device)

        # Rasterize bounding box images for reconstruction
        targets_img = make_bounding_box_images(targets)
        targets_img = targets_img.to(device)

        # Make the targets for the Segmentation Network
        targets_seg = convert_bounding_box_targets(targets, device)

        road_map = torch.stack(road_map).float()
        road_map = road_map.view(-1, 1, 800, 800)
        road_map = road_map.to(device)

        # print('input shape:', images.shape)
        # print('targt shape:', road_map.shape)

        # obj_reconstructions, obj_mu, obj_logvar = model(images, mode='object-map')
        # road_reconstructions, road_mu, road_logvar = model(images, mode='road-map')

        obj_recon, road_recon, mu, logvar = model(images, mode='object-road-maps')

        seg_losses = seg_model(obj_recon, targets_seg)
        
        #print(seg_losses)
        # print('outpt shape:', reconstructions.shape)

        road_loss, road_bce, road_kld = criterion(road_recon, road_map,
                                                  mode='road-map',
                                                  mu=mu,
                                                  logvar=logvar,
                                                  kld_schedule=0.05,
                                                  i=i)
        obj_loss, obj_bce, obj_kld = criterion(obj_recon, targets_img,
                                               mode='object-map',
                                               mu=mu,
                                               logvar=logvar,
                                               kld_schedule=0.05,
                                               i=i)

        loss = road_loss + obj_loss + seg_losses['loss_box_reg'] + seg_losses['loss_rpn_box_reg']
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
    torch.save(seg_model.state_dict(), 'segmentation-network-latest.torch')

print('Labeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))
