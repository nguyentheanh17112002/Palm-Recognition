import torch
from thop import profile
from lightning import LightningModule

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.ArcPalmModule import ArcPalmModule

if __name__ ==  "__main__":
    path = "/home/anhnt596/Palm-Recognition/logs/train/runs/Thesis_VisionTransformer_ArcFaceLoss_Canny/checkpoints/epoch_071.ckpt"
    arcpalm_module = ArcPalmModule.load_from_checkpoint(path)

    model = arcpalm_module.backbone.eval()

    input = torch.rand(1,6,224,224, device='cuda')


    macs, params = profile(model, inputs=(input, ))
                           
    print("macs: ", macs)
    print("params: ", params)