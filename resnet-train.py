#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Parameters

# In[ ]:


batch_size = 4
hidden_size = 1024

unlabeled_epochs = 10
labeled_epochs = 100

# # Setup

# In[3]:


from time import perf_counter

import torch

from data import get_unlabeled_set, get_labeled_set, set_seeds, make_bounding_box_images
from model.resnet import Prototype

# In[4]:


set_seeds()
torch.backends.cudnn.benchmark = True

# In[5]:


if torch.cuda.is_available():
    batch_size *= torch.cuda.device_count()

_, unlabeled_trainloader = get_unlabeled_set(batch_size=batch_size)
(_, labeled_trainloader), (_, labeled_testloader) = get_labeled_set(batch_size=batch_size, validation=0.2)

# In[6]:


# import matplotlib.pyplot as plt
# from helpers.helper import draw_box
# # The center of image is 400 * 400
# fig, ax = plt.subplots()
# color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
# ax.imshow(road_image[0], cmap ='binary');
# # The ego car position
# ax.plot(400, 400, 'x', color="red")
# for i, bb in enumerate(target[0]['bounding_box']):
#     # You can check the implementation of the draw box to understand how it works
#     draw_box(ax, bb, color=color_list[target[0]['category'][i]])


# # Model

# In[7]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[8]:


model = Prototype(hidden_dim=hidden_size)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    assert batch_size >= torch.cuda.device_count()

model = model.to(device)

# ### Unlabeled Pre-training

# In[9]:


criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())

# In[10]:


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

    model.save(file_prefix='unlabeled-')

print('Unlabeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))

# ### Labeled Training

# In[11]:


model.train()

start_time = perf_counter()
for epoch in range(labeled_epochs):
    loss = 0.0

    max_batches = len(labeled_trainloader)
    for idx, (images, targets, _) in enumerate(labeled_trainloader):
        optimizer.zero_grad()

        # Restack images to be (6-directions, batch size, channels, H, W)
        images = torch.stack(images).to(device).permute(1, 0, 2, 3, 4)

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

    model.save(file_prefix='labeled-')

print('Labeled Training Took (Min):', round(int(perf_counter() - start_time) / 60, 2))

# In[12]:


# tensor_to_image(reconstructions[0].detach(), 2, 'uint8')
