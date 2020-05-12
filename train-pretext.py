import os
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

from data import get_unlabeled_set, horz_flip_tensor
from model.resnet_fpn import ResnetFPN
from model.util import Flatten

#
# Constants
#

output_path = '/scratch/css459/fpn_weights/'

unlabeled_epochs = 4
unlabeled_batch_size = 8

#
# Setup
#

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        unlabeled_batch_size *= torch.cuda.device_count()


def save(m, file_name):
    file_path = os.path.join(output_path, file_name)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.save(m.module.state_dict(), file_path)
    else:
        torch.save(m.state_dict(), file_path)


def load(m, file_name):
    file_path = os.path.join(output_path, file_name)
    m.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))


#
# Data
#

_, unlabeled_trainloader = get_unlabeled_set(batch_size=unlabeled_batch_size,
                                             format='sample')

#
# Model
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = ResnetFPN()
img_seq_predictor = nn.Sequential(
    nn.Conv2d(10, 3, 3),
    nn.LeakyReLU(inplace=True),

    nn.Conv2d(3, 1, 3),
    nn.AdaptiveMaxPool2d((32, 32)),
    nn.LeakyReLU(inplace=True),

    Flatten(),

    nn.Linear(1024, 256),
    nn.LeakyReLU(inplace=True),
    nn.Linear(256, 64),
    nn.LeakyReLU(inplace=True),
    nn.Linear(64, 36)
)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    encoder = torch.nn.DataParallel(encoder)
    img_seq_predictor = torch.nn.DataParallel(img_seq_predictor)

    assert unlabeled_batch_size >= torch.cuda.device_count()
    print('==> Using Data Parallel Devices:', torch.cuda.device_count())

encoder = encoder.to(device)
img_seq_predictor = img_seq_predictor.to(device)

print('==> Device:', device)
print('==> Batch Size:', unlabeled_batch_size)
print('==> Unlabeled Epochs:', unlabeled_epochs)
print('==> Model Loaded. Begin Training')

#
# Loss
#

criterion = nn.CrossEntropyLoss()
optimizer_encoder = torch.optim.Adam(encoder.parameters())
optimizer_decoder = torch.optim.Adam(img_seq_predictor.parameters())

encoder.train()
img_seq_predictor.train()

i = 0
start_time = perf_counter()
for epoch in range(unlabeled_epochs):

    loss = 0.0
    max_batches = len(unlabeled_trainloader)
    for idx, images in enumerate(unlabeled_trainloader):
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        # Format inputs

        # Flip the back images along their height,
        # this makes more sense to orient the feature maps
        images_fmt = []
        orderings = []
        for batch in images:
            # Random Shuffle
            ordering = list(range(6))
            np.random.shuffle(ordering)
            batch = [batch[i] for i in ordering]

            t = torch.stack(list(batch[:3]) + horz_flip_tensor(batch[-3:])).float()
            images_fmt.append(t)
            orderings.append(ordering)

        images = torch.stack(images_fmt)
        orderings = torch.from_numpy(np.array(orderings))
        images = images.to(device)
        orderings = orderings.to(device)

        # Forward
        enc = encoder(images)
        pred_ordering = img_seq_predictor(enc)
        pred_ordering = pred_ordering.view(-1, 6, 6)

        # Losses
        loss = criterion(pred_ordering, orderings)
        loss.backward()

        optimizer_encoder.step()
        optimizer_decoder.step()

        i += 1

        if idx % 100 == 0:
            print('[', epoch, '|', idx, '\t/', max_batches, ']', 'loss:', round(loss.item(), 4),
                  'curr time mins:', round(int(perf_counter() - start_time) / 60, 2))

    save(encoder, 'encoder-pretext-latest.torch')
