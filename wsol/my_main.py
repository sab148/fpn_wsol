from torch._C import memory_format
from detectron2.config import get_cfg
import fpn

from layers import ShapeSpec
from detectron2 import model_zoo
from detectron2.config import get_cfg
import torch 

if __name__ == '__main__':
    
    cfg = model_zoo.get_config("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
    model = fpn.build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
    input = torch.randn(1, 3, 256, 256)
    print(model(input)['p2'].size())
    print(model(input)['p3'].size())
    print(model(input)['p4'].size())
    print(model(input)['p5'].size())
    print(model(input)['p6'].size())