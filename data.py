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

        print('==> Validation Index:', validation_idx)

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

        return (labeled_train_set, labeled_train_loader), (labeled_test_set, labeled_test_loader)


def make_bounding_box_images(batch, single_channel=True):
    return torch.stack([_make_bounding_box_img_simple(b, single_channel=single_channel)
                        for b in batch])

def _make_bounding_box_img_simple(sample, single_channel):
    boxes = []
    b_boxes = sample['bounding_box']

    # Iterate over boxes
    for i in range(len(b_boxes)):
        b = b_boxes[i]
        b = b.T * 10
        b[:, 1] *= -1
        b += 400
        b = [tuple(x) for x in b.numpy()]
        b[-2], b[-1] = b[-1], b[-2]
        boxes.append(b)

    # All Mask
    car_mask = Image.new('1', (800, 800))
    context = ImageDraw.Draw(car_mask)
    for b in boxes:
        context.polygon(b, fill=1)

    car_mask = np.array(car_mask).astype(np.float)
    return torch.from_numpy(car_mask).float()[None, ...]

def _make_bounding_box_img_helper(sample, single_channel):
    boxes = []
    categories = []

    b_boxes = sample['bounding_box']
    cat = sample['category']

    # Iterate over boxes
    for i in range(len(b_boxes)):
        b = b_boxes[i]
        b = b.T * 10
        b[:, 1] *= -1
        b += 400
        b = [tuple(x) for x in b.numpy()]
        b[-2], b[-1] = b[-1], b[-2]
        c = cat[i].item() + 1  # Categories incremented by 1
        boxes.append(b)
        categories.append(c)

    # Build image
    channels = []

    # Categories incremented by 1
    car_index = 2 + 1

    # Car Mask
    car_mask = Image.new('1', (800, 800))
    context = ImageDraw.Draw(car_mask)
    boxes_idx = [i for i in range(len(categories)) if categories[i] == car_index]
    for i in boxes_idx:
        context.polygon(boxes[i], fill=1)
    channels.append(np.array(car_mask).astype(np.float32))

    # Other Mask
    other_mask = Image.new('1', (800, 800))
    context = ImageDraw.Draw(other_mask)
    boxes_idx = [i for i in range(len(categories)) if categories[i] != car_index]
    for i in boxes_idx:
        context.polygon(boxes[i], fill=1)
    channels.append(np.array(other_mask).astype(np.float32))

    # We only need to predict: "background", "car", "sometihng else"
    # for c in range(1, 10):
    #    canvas = Image.new('1', (800, 800))
    #    context = ImageDraw.Draw(canvas)
    #    boxes_idx = [i for i in range(len(categories)) if categories[i] == c]
    #    for i in boxes_idx:
    #        context.polygon(boxes[i], fill=1)
    #    channels.append(np.array(canvas).astype(np.float32))

    # Background mask
    mask = np.logical_not(np.sum(np.array(channels), 0, dtype=np.float32))

    # Stack masks along channels
    channels = np.array([mask] + channels)

    if single_channel:
        channels = np.array([i * channels[i] for i in range(len(channels))]).astype(np.long)
        channels = np.max(channels, 0)

    return torch.from_numpy(channels)


# Required for FasterRCNN PyTorch implementation
# to convert the given format to its format
def convert_bounding_box_targets(targets, device):
    return [_convert_bounding_box_targets_helper(t, device) for t in targets]


def _convert_bounding_box_targets_helper(sample, device):
    boxes = []
    categories = []
    b_boxes = sample['bounding_box'].numpy()
    cat = sample['category']

    # Iterate over boxes
    for i in range(len(b_boxes)):
        b = b_boxes[i]
        b = b.T * 10
        b[:, 1] *= -1
        b += 400

        x_min = min([x[0] for x in b])
        x_max = max([x[0] for x in b])
        y_min = min([y[1] for y in b])
        y_max = max([y[1] for y in b])

        b = [x_min, y_min, x_max, y_max]
        c = cat[i].item() + 1  # Categories incremented by 1
        boxes.append(b)
        categories.append(c)

    # Anything that's not a car is something else
    car_index = 2 + 1
    #categories = [1 if c == car_index else 2 for c in categories]
    categories = [1] * len(categories)

    # Make tensor
    boxes = torch.FloatTensor(boxes).to(device)
    categories = torch.LongTensor(categories).to(device)

    return {'boxes': boxes, 'labels': categories}


def convert_bounding_box_inference(preds):
    return [_convert_bounding_box_inference_helper(p) for p in preds]


def _convert_bounding_box_inference_helper(pred):
    boxes = pred['boxes']
    new_boxes = []
    for b in boxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        new_boxes.append([[x2, x2, x1, x1], [y1, y2, y1, y2]])

    return torch.DoubleTensor(new_boxes).to('cuda')  # HACK HACK HACK


def tensor_to_image(x, channel=0, as_type='bool', detach=False):
    c = x[channel]
    if detach:
        c = c.detach()
    return Image.fromarray(c.numpy().astype(as_type)).convert('1')
