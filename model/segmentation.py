from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.resnet import resnet18

from data import convert_bounding_box_inference
from model.util import remove_backbone_head


class SegmentationNetwork(nn.Module):
    def __init__(self, backbone=None, output_channels=2, backbone_output_channels=512):
        super().__init__()

        if not backbone:
            b, _ = remove_backbone_head(resnet18(pretrained=False))
            backbone = b

        # ResNet produces 512-length outputs
        backbone.out_channels = backbone_output_channels
        self.segmentation_network = FasterRCNN(backbone, num_classes=output_channels)

    def forward(self, x, boxes=None):
        x = self.segmentation_network(x, boxes)
        return x

    def infer(self, x):
        self.eval()
        self.segmentation_network.eval()
        x = self.segmentation_network(x)
        return convert_bounding_box_inference(x)
