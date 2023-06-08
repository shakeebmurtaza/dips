from dlib_extended.unet_with_class_head.decoder import UnetDecoder
from dlib_extended.unet_with_class_head.decoder import UnetFCAMDecoder
from dlib_extended.encoders import get_encoder
from dlib_extended.base_with_class_head import SegmentationModel
from dlib_extended.base_with_class_head import FCAMModel
from dlib_extended.base_with_class_head import SegmentationHead
from dlib_extended.base_with_class_head import ClassificationHead
from dlib_extended.base_with_class_head import ReconstructionHead
from dlib_extended.configure import constants
from typing import Optional, Union, List
import numpy as np
import torch

class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic
    segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract
    features of different spatial
    resolution (skip connections) which are used by decoder to define accurate
    segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.
    Args:
        encoder_name: Name of the classification model that will be used as an
            encoder (a.k.a backbone) to extract features of different spatial
            resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
        Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for
            depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W),
            (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization),
        **"imagenet"** (pre-training on ImageNet) and other pretrained
        weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels**
            parameter for convolutions used in decoder. Length of the list
            should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D
            and Activation layers is used. If **"inplace"** InplaceABN will
            be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask (or you can think as a
            number of channels of output mask). Useful ONLY for the task
            constants.SEG.
        activation: An activation function to apply after the final convolution
            layer.
            Available options are **"sigmoid"**, **"softmax"**,
            **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and
            **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply
                "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: Unet
    .. _Unet:
        https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        scale_in: float = 1.,
        classification_head_classes: int = 2
    ):
        super().__init__()

        self.task = constants.SEG
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        self.x_in = None

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.classification_head = ClassificationHead(in_channels=decoder_channels[-1], classes=classification_head_classes)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

        for m in self.classification_head.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

# net = Unet(encoder_name='resnet50', in_channels=6, classes=2)
# output = net(torch.tensor(np.zeros((1,6,224,224))).float())
# print(output.shape)