from time import perf_counter

import numpy as np
import torch

from data import get_unlabeled_set, get_labeled_set, make_bounding_box_images
from model.resnet_fpn import ResnetFPN, MapReconstructor

#
# Constants
#

output_path = '../fpn_weights/'

unlabeled_batch_size = 32
labeled_batch_size = 4

unlabeled_epochs = 10
labeled_epochs = 30

#
# Setup
#

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
encoder = ResnetFPN()
obj_decoder = MapReconstructor(10, output_channels=2)
road_decoder = MapReconstructor(10, output_channels=2)

encoder = encoder.to(device)
obj_decoder = obj_decoder.to(device)
road_decoder = road_decoder.to(device)

print('==> Device:', device)
print('==> Batch Size:', unlabeled_batch_size)
print('==> Unlabeled Epochs:', unlabeled_epochs)
print('==> Labeled Epochs:', labeled_epochs)
print('==> Model Loaded. Begin Training')

#
# Loss
#

class_weights = torch.FloatTensor([0.03, 0.97]).to(device)
road_weights = torch.FloatTensor([0.3, 0.70]).to(device)
criterion_obj = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
criterion_road = torch.nn.BCEWithLogitsLoss(pos_weight=road_weights)

optimizer_encoder = torch.optim.Adam(encoder.parameters())
optimizer_decoder_obj = torch.optim.Adam(obj_decoder.parameters())
optimizer_decoder_road = torch.optim.Adam(road_decoder.parameters())

encoder.train()
obj_decoder.train()
road_decoder.train()

i = 0
start_time = perf_counter()
for epoch in range(labeled_epochs):

    loss = 0.0
    max_batches = len(labeled_trainloader)
    for idx, (images, targets, road_map) in enumerate(labeled_trainloader):
        optimizer_encoder.zero_grad()
        optimizer_decoder_obj.zero_grad()
        optimizer_decoder_road.zero_grad()

        # Format inputs
        images = torch.stack(images).float()
        images = images.to(device)

        # Rasterize bounding box images for reconstruction
        obj_map = make_bounding_box_images(targets).float()
        obj_map = obj_map.to(device)

        # Format road map
        road_map = torch.stack(road_map).float()
        road_map = [np.logical_not(road_map).float(), road_map]
        road_map = torch.stack(road_map).permute(1, 0, 2, 3)
        road_map = road_map.to(device)

        # Forward
        enc = encoder(images)
        road_map_recon = road_decoder(enc)
        obj_map_recon = obj_decoder(enc)

        # Permute the dims for PyTorch BCELoss
        road_map_recon = road_map_recon.permute(0, 2, 3, 1)
        obj_map_recon = obj_map_recon.permute(0, 2, 3, 1)
        obj_map = obj_map.permute(0, 2, 3, 1)
        road_map = road_map.permute(0, 2, 3, 1)
        
        #print(road_map_recon.shape)
        #print(road_map.shape)
        #print(obj_map.shape)
        #print(obj_map_recon.shape)

        # Losses
        loss_obj = criterion_obj(obj_map_recon, obj_map)
        loss_road = criterion_road(road_map_recon, road_map)

        loss = loss_obj + loss_road
        loss.backward()

        optimizer_encoder.step()
        optimizer_decoder_road.step()
        optimizer_decoder_obj.step()

        i += 1

        # Training Wheels
        # print('loss', loss.item())
        # break
        if idx % 100 == 0:
            print('[', epoch, '|', idx, '/', max_batches, ']', 'loss:', loss.item(),
                  'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))
