from torch import nn

from model.util import make_object_seg_network_from_backbone


class SegmentationNetwork(nn.Module):
    def __init__(self, backbone, output_channels):
        super().__init__()
        self.segmentation_network = make_object_seg_network_from_backbone(backbone,
                                                                          512, output_channels)

    def forward(self, x, boxes=None):
        x = self.segmentation_network(x, boxes)
