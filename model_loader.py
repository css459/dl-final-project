"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import os
import torch
import torchvision
from model.resnet import Prototype
from model.segmentation import SegmentationNetwork

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1():
    return torchvision.transforms.ToTensor()


# For road map task
def get_transform_task2():
    return torchvision.transforms.ToTensor()


class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    team_number = 1
    round_number = 1
    team_member = []
    contact_email = '@nyu.edu'

    def __init__(self, model_file='put_your_model_file(or files)_name_here'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Prototype(device, hidden_dim=1024, variational=True)

        unlabeled_backbone = 'unlabeled-backbone-latest.torch'
        labeled_backbone = 'labeled-roadmap-backbone-latest.torch'

        model.load_backbone(unlabeled_backbone)
        model.load_state_dict(torch.load(labeled_backbone))

        self.model = model.to(device)

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        self.model.eval()

        with torch.no_grad():
            obj_recon, road_recon, mu, logvar = self.model(samples, mode='object-road-maps')
            return obj_recon
