import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib_extended.base_with_class_head.model import SegmentationModel
from dlib_extended.base_with_class_head.model import STDClModel
from dlib_extended.base_with_class_head.model import FCAMModel


from dlib_extended.base_with_class_head.modules import (
    Conv2dReLU,
    Attention,
)

from dlib_extended.base_with_class_head.heads import (
    SegmentationHead,
    ClassificationHead,
    ReconstructionHead
)
