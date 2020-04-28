import random

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

from helpers.data_helper import UnlabeledDataset, LabeledDataset
from helpers.helper import collate_fn


def set_seeds(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


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


def tensor_to_image(x, channel=0):
    c = x[channel]
    return Image.fromarray(c.numpy().astype('bool')).convert('1')
