import random

import numpy as np
import torch
import torchvision

from helpers.data_helper import UnlabeledDataset, LabeledDataset
from helpers.helper import collate_fn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

image_folder = '../data'
annotation_csv = '../data/annotation.csv'

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)

# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

transform = torchvision.transforms.ToTensor()


def get_unlabeled_set(batch_size=3, format='image'):
    assert format in ['image', 'sample']

    unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                          scene_index=unlabeled_scene_index,
                                          first_dim='image',
                                          transform=transform)

    unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=2)
    return unlabeled_trainset, unlabeled_trainloader


def get_labeled_set(batch_size=3):
    # The labeled dataset can only be retrieved by sample.
    # And all the returned data are tuple of tensors, since bounding boxes may have different size
    # You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform,
                                      extra_info=False)

    labeled_trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      collate_fn=collate_fn)

    return labeled_trainset, labeled_trainloader
