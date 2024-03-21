from typing import Any, List, Dict, Tuple
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from lightning import LightningModule


from torchmetrics import MaxMetric, MeanMetric, AUROC, ROC
from torchmetrics.classification.accuracy import Accuracy

from pytorch_metric_learning import losses

import wandb
import pandas as pd

import numpy as np

import numpy as np

from sklearn import metrics


import torch.nn.functional as F

class ArcPalmModule(LightningModule):
    def __init__(
                self,
                backbone: torch.nn.Module,
                loss: str,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                compile: bool,
                num_classes: int = 1527, 
                pretrain_backbone: str = None
                ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.num_classes = num_classes
        if pretrain_backbone is not None:
            self.backbone = ArcPalmModule.load_from_checkpoint(pretrain_backbone).backbone.train()
        else:
            self.backbone = backbone
        
        if loss == "ArcFaceLoss":
            self.loss = losses.ArcFaceLoss(num_classes=self.num_classes, embedding_size=512, margin=30.0, scale=48)
        elif loss == "CosFaceLoss":
            self.loss = losses.CosFaceLoss(num_classes=self.num_classes, embedding_size =512, margin=0.35, scale=64)
        elif loss == "SubCenterArcFaceLoss":
            self.loss = losses.SubCenterArcFaceLoss(num_classes= self.num_classes, embedding_size = 512, margin=28.6, scale=48, sub_centers=3)
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.test_preds = torch.empty(0, device='cuda')
        self.test_targets = torch.empty(0, device='cuda')

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        x, y = batch
        features = self.forward(x)
        loss = self.loss(features, y)
        logits = self.loss.get_logits(features)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
    
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor ], batch_idx: int):
        img1 , img2 , targets = batch
        feature1 = self.forward(img1)
        feature2 = self.forward(img2)

        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        
        score = torch.sum(feature1*feature2, dim=1)

        self.test_preds = torch.cat((self.test_preds, score), dim = 0)
        self.test_targets = torch.cat((self.test_targets, targets), dim = 0)


    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        fpr, tpr, thresholds = metrics.roc_curve(self.test_targets.cpu().numpy(), self.test_preds.cpu().numpy())

        data = {
            "TPR": tpr,
            "FPR": fpr,
            "Thresholds": thresholds
        }

        df = pd.DataFrame(data)

        row_000001 = df[df['FPR'] > 0.000001].iloc[0]

        row_00001 = df[df['FPR'] > 0.00001].iloc[0]

        row_0001 = df[df['FPR'] > 0.0001].iloc[0]

        row_001 = df[df['FPR'] > 0.001].iloc[0]

        row_01 = df[df['FPR'] > 0.01].iloc[0]

        tbl = wandb.Table(dataframe=pd.DataFrame([row_000001, row_00001, row_0001, row_001, row_01]))
        wandb.log({"TPR/FPR" : tbl})

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.
)
        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.backbone = torch.compile(self.backbone)
            self.loss = torch.compile(self.loss)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ArcPalmModule(None, None, None, None, None)