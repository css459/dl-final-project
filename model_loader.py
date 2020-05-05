"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

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
    team_name = 'Console Cowboys'
    team_number = 999
    round_number = 2
    team_member = ['Cole Smith']
    contact_email = 'css@nyu.edu'

    def __init__(self, model_file='resvar.torch', model_file2='segmenter.torch'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Prototype(device, hidden_dim=1024, variational=True)
        self.seg_model = SegmentationNetwork(self.model.backbone, 3)

        # Load models
        self.model.load_state_dict(torch.load(model_file))
        self.seg_model.load_state_dict(torch.load(model_file2))

        self.model.eval()
        self.seg_model.eval()

        self.model.to(device)
        self.seg_model.to(device)

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        outputs = self.seg_model.infer(self.model.infer_object_heat_map(samples))
        print(outputs.shape)
        return outputs
        # return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        outputs = self.model.infer_road_map(samples)
        return outputs
        # return torch.rand(1, 800, 800) > 0.5
