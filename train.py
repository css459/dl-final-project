import os
from time import perf_counter

import numpy as np
import torch

from data import get_unlabeled_set, get_labeled_set, \
    make_bounding_box_images, horz_flip_tensor, convert_bounding_box_targets
from model.resnet_fpn import ResnetFPN, MapReconstructor
from model.segmentation import SegmentationNetwork

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

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    unlabeled_batch_size *= torch.cuda.device_count()
    labeled_batch_size *= torch.cuda.device_count()


def save(m, file_name):
    file_path = os.path.join(output_path, file_name)
    if torch.cuda.is_available():
        torch.save(m.module.state_dict(), file_path)
    else:
        torch.save(m.state_dict(), file_path)


def load(m, file_name):
    file_path = os.path.join(output_path, file_name)
    return m.load_state_dict(torch.load(file_path))


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
seg_model = SegmentationNetwork()

if torch.cuda.is_available():
    encoder = torch.nn.DataParallel(encoder)
    obj_decoder = torch.nn.DataParallel(obj_decoder)
    road_decoder = torch.nn.DataParallel(road_decoder)
    seg_model = torch.nn.DataParallel(seg_model)

    assert unlabeled_batch_size >= torch.cuda.device_count()
    print('==> Using Data Parallel Devices:', torch.cuda.device_count())

encoder = encoder.to(device)
obj_decoder = obj_decoder.to(device)
road_decoder = road_decoder.to(device)
seg_model = seg_model.to(device)

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
optimizer_seg_model = torch.optim.Adam(seg_model.parameters())

encoder.train()
obj_decoder.train()
road_decoder.train()
seg_model.train()

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

        # Flip the back images along their height,
        # this makes more sense to orient the feature maps
        images_fmt = []
        for batch in images:
            t = torch.stack(list(batch[:3]) + horz_flip_tensor(batch[-3:])).float()
            images_fmt.append(t)

        images = torch.stack(images_fmt)
        images = images.to(device)

        # Rasterize bounding box images for reconstruction
        obj_map = make_bounding_box_images(targets).float()
        obj_map = obj_map.to(device)

        # Format road map
        road_map = torch.stack(road_map).float()
        road_map = [np.logical_not(road_map).float(), road_map]
        road_map = torch.stack(road_map).permute(1, 0, 2, 3)
        road_map = road_map.to(device)

        # Make the targets for the Segmentation Network
        targets_seg = convert_bounding_box_targets(targets, device)

        # Forward
        enc = encoder(images)
        road_map_recon = road_decoder(enc)
        obj_map_recon = obj_decoder(enc)
        seg_losses = seg_model(torch.sigmoid(obj_map_recon[:, 1, :, :].view(-1, 1, 800, 800)), targets_seg)

        # Permute the dims for PyTorch BCELoss
        road_map_recon = road_map_recon.permute(0, 2, 3, 1)
        obj_map_recon = obj_map_recon.permute(0, 2, 3, 1)
        obj_map = obj_map.permute(0, 2, 3, 1)
        road_map = road_map.permute(0, 2, 3, 1)

        # Losses
        loss_obj = criterion_obj(obj_map_recon, obj_map)
        loss_road = criterion_road(road_map_recon, road_map)
        loss_seg = seg_losses['loss_box_reg'] + seg_losses['loss_rpn_box_reg']

        loss = loss_obj + loss_road + loss_seg
        loss.backward()

        optimizer_encoder.step()
        optimizer_decoder_road.step()
        optimizer_decoder_obj.step()
        optimizer_seg_model.step()

        i += 1

        # Training Wheels
        # print('loss', loss.item())
        # break
        if idx % 100 == 0:
            print('[', epoch, '|', idx, '\t/', max_batches, ']', 'loss:', round(loss.item(), 4),
                  '( obj:', round(loss_obj.item(), 4), 'road:', round(loss_road.item(), 4),
                  'seg:', round(loss_seg.item(), 4), ') curr time mins:',
                  round(int(perf_counter() - start_time) / 60, 2))

    save(encoder, 'encoder-latest.torch')
    save(obj_decoder, 'obj-decoder-latest.torch')
    save(road_decoder, 'road-decoder-latest.torch')
    save(seg_model, 'seg-latest.torch')
