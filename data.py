import random

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from helpers.data_helper import UnlabeledDataset, LabeledDataset
from helpers.helper import collate_fn


def set_seeds(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


#
# Data Constants
#

IMAGE_FOLDER = '../data'
ANNOTATION_CSV = '../data/annotation.csv'

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
UNLABELED_SCENE_INDEX = np.arange(106)

# The scenes from 106 - 133 are labeled
# You should divide the labeled_scene_index into two subsets (training and validation)
LABELED_SCENE_INDEX = np.arange(106, 134)

#
# PyTorch Data Transformers
#

transform = torchvision.transforms.ToTensor()


def get_unlabeled_set(batch_size=3, format='image'):
    assert format in ['image', 'sample']

    unlabeled_trainset = UnlabeledDataset(image_folder=IMAGE_FOLDER,
                                          scene_index=UNLABELED_SCENE_INDEX,
                                          first_dim='image',
                                          transform=transform)

    unlabeled_trainloader = DataLoader(unlabeled_trainset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)

    return unlabeled_trainset, unlabeled_trainloader


def get_labeled_set(batch_size=3, validation=None, extra_info=False):
    if not validation:
        labeled_train_set = LabeledDataset(image_folder=IMAGE_FOLDER,
                                           annotation_file=ANNOTATION_CSV,
                                           scene_index=LABELED_SCENE_INDEX,
                                           transform=transform,
                                           extra_info=extra_info)

        labeled_train_loader = DataLoader(labeled_train_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=collate_fn)
        return labeled_train_set, labeled_train_loader

    else:
        labeled_sample_size = LABELED_SCENE_INDEX[-1] - LABELED_SCENE_INDEX[0]
        validation_idx = LABELED_SCENE_INDEX[-1] - int(labeled_sample_size * validation)
        assert validation_idx > LABELED_SCENE_INDEX[0]

        print('Validation Index:', validation_idx)

        train_scene_idx = np.arange(LABELED_SCENE_INDEX[0], validation_idx)
        test_scene_idx = np.arange(validation_idx, LABELED_SCENE_INDEX[-1])

        labeled_train_set = LabeledDataset(image_folder=IMAGE_FOLDER,
                                           annotation_file=ANNOTATION_CSV,
                                           scene_index=train_scene_idx,
                                           transform=transform,
                                           extra_info=extra_info)

        labeled_train_loader = DataLoader(labeled_train_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=collate_fn)

        labeled_test_set = LabeledDataset(image_folder=IMAGE_FOLDER,
                                          annotation_file=ANNOTATION_CSV,
                                          scene_index=test_scene_idx,
                                          transform=transform,
                                          extra_info=extra_info)

        labeled_test_loader = DataLoader(labeled_test_set,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         collate_fn=collate_fn)

        return (labeled_train_set, labeled_train_loader), (labeled_test_set, labeled_train_loader)


def make_bounding_box_images(batch):
    return torch.stack([_make_bounding_box_img_helper(b)
                        for b in batch])


def _make_bounding_box_img_helper(sample):
    boxes = []
    categories = []

    b_boxes = sample['bounding_box']
    cat = sample['category']

    # Iterate over boxes
    for i in range(len(b_boxes)):
        b = b_boxes[i]
        b = b.T * 10
        b[:, 1] *= -1
        b += + 400
        b = [tuple(x) for x in b.numpy()]
        b[-2], b[-1] = b[-1], b[-2]
        c = cat[i].item() + 1
        boxes.append(b)
        categories.append(c)

    # Build image
    channels = []
    for c in range(1, 10):
        canvas = Image.new('1', (800, 800))
        context = ImageDraw.Draw(canvas)
        boxes_idx = [i for i in range(len(categories)) if categories[i] == c]
        for i in boxes_idx:
            context.polygon(boxes[i], fill=1)
        channels.append(np.array(canvas).astype(float))

    # Background mask
    mask = np.logical_not(np.sum(np.array(channels), 0, dtype=np.float))
    channels = np.array([mask] + channels)

    return torch.from_numpy(channels)


def tensor_to_image(x, channel=0, astype='bool'):
    c = x[channel]
    return Image.fromarray(c.numpy().astype(astype)).convert('1')
