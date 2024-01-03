from typing import Any, List, Tuple

import torch
import os
import sys
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .COEP_Dataset import COEP_Dataset

class COEP_DataModule(LightningDataModule):
    def __init__(self,
                 root_dir: str = './database',
                 batch_size: int = 64,
                 train_val_split: List = [1040, 265],
                 num_workers: int = 0,) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers

    def setup(self, stage = None) -> None:
        if stage == 'fit' or stage is None:
            self.dataset = COEP_Dataset(self.root_dir)
            self.train_set, self.val_set = random_split(self.dataset, self.train_val_split)
            
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
if __name__ == "__main__":
    _ = COEP_DataModule("/home/anhnt596/PalmPrint/database")