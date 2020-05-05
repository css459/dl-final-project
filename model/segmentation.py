from torch import nn
from torchvision.models.detection import FasterRCNN

from data import convert_bounding_box_inference


class SegmentationNetwork(nn.Module):
    def __init__(self, backbone, output_channels, backbone_output_channels=512):
        super().__init__()

        # ResNet produces 512-length outputs
        backbone.out_channels = backbone_output_channels

        # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        #                                    aspect_ratios=((0.5, 1.0, 2.0),))
        #
        # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
        #                                                 output_size=7,
        #                                                 sampling_ratio=2)
        # self.segmentation_network = make_object_seg_network_from_backbone(backbone,
        #                                                                   512, output_channels)

        self.segmentation_network = FasterRCNN(backbone, num_classes=output_channels)

    def forward(self, x, boxes=None):
        x = self.segmentation_network(x, boxes)
        return x

    def infer(self, x):
        self.eval()
        self.segmentation_network.eval()
        x = self.segmentation_network(x)
        return convert_bounding_box_inference(x)
