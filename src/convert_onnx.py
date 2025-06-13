import torch
import lightning
import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from models import ArcPalmModule

if __name__ == "__main__":
    path = "/home/anhnt596/Palm-Recognition/logs/train/runs/2024-01-18_10-58-28/checkpoints/epoch_141.ckpt"
    arcpalm_module = ArcPalmModule.load_from_checkpoint(path)

    model = arcpalm_module.backbone.eval()

    batch_size = 1

    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True, device = 'cuda')

    device = 'cuda'
    model.to(device)

    torch.onnx.export(
        model,
        x,
        "/home/anhnt596/Palm-Recognition/onnx/Resnet50.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'output' : {0 : 'batch_size'}}
    )
    print("Done")