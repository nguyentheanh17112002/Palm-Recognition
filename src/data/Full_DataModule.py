from typing import Any, List, Tuple
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import torch
import os
import sys
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .Full_Dataset import Full_Dataset

class Full_DataModule(LightningDataModule):
    def __init__(self,
                 root_dir: str = './/home/anhnt596/Palm-Recognition/data/Full',
                 num_classes: int = 1581,
                 batch_size: int = 64,
                 train_val_split: List = [85, 15],
                 random_seed:int = 42,
                 test_size: int = 100000,
                 num_workers: int = 0,) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.test_size = test_size
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers

    def setup(self, stage = None) -> None:
        if stage == 'fit' or stage is None:
            self.dataset = Full_Dataset(self.root_dir, train=True)
            train_size = int(len(self.dataset)*self.train_val_split[0]/100)
            val_size = len(self.dataset)-train_size
            self.train_set, self.val_set = random_split(self.dataset, [train_size, val_size])
        elif stage == 'test':
            self.test_set = Full_Dataset(self.root_dir, train=False, random_seed=self.random_seed, test_size=self.test_size) 
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
if __name__ == "__main__":
    coep = Full_DataModule("/home/anhnt596/Palm-Recognition/data")
    coep.setup(stage='fit')

    trainloader = coep.train_dataloader()
    coep.setup(stage='test')
    testloader = coep.test_dataloader()

    for batch in trainloader:
        x, y = batch
        print(x.shape)
        print(y.shape)

        break

    for batch in testloader:
        x,y,z = batch
        print(x.shape)
        print(y.shape)
        print(z.shape)
        break