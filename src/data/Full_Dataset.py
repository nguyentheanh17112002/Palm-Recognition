import os
import random
import re
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CustomTransform(object):
    def __call__(self, img):
        img_np = np.array(img)

        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        edge1 = cv2.Canny(gray_img, 10, 20)
        edge2 = cv2.Canny(gray_img, 20, 40)
        edge3 = cv2.Canny(gray_img, 40, 80)

        # Mở rộng kích thước của mỗi cạnh từ (H, W) thành (H, W, 1) để có 1 channel
        edge1 = np.expand_dims(edge1, axis=2)
        edge2 = np.expand_dims(edge2, axis=2)
        edge3 = np.expand_dims(edge3, axis=2)

        img_np = np.concatenate((img_np, edge1, edge2, edge3), axis=2)

        return img_np


class Full_Dataset(Dataset):
    def __init__(self, root_dir: str, type_data: str, train: bool) -> None:
        super().__init__()
        self.train = train
        self.root_dir = root_dir
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "test")

        self.pair_path = os.path.join(root_dir, "test_pairs.txt")
        if type_data == "base":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif type_data == "canny":
            self.transform = transforms.Compose(
                [
                    CustomTransform(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
                        std=[0.229, 0.224, 0.225, 0.5, 0.5, 0.5],
                    ),
                ]
            )

        self.train_set = ImageFolder(self.train_dir, transform=self.transform)
        self.test_pair = self._create_pair()

    def _create_pair(self):
        with open(self.pair_path, "r") as file:
            lines = file.readlines()

        result_list = []

        for line in lines:
            elements = line.strip().split(",")
            if elements[2].isdigit():
                elements[2] = int(elements[2])
            result_list.append(tuple(elements))

        return result_list

    def __len__(self):
        if self.train:
            return len(self.train_set)
        else:
            return len(self.test_pair)

    def __getitem__(self, index):
        if self.train:
            return self.train_set[index]
        else:
            path1, path2, val = self.test_pair[index]
            path1 = os.path.join(self.test_dir, path1)
            path2 = os.path.join(self.test_dir, path2)

            img1 = Image.open(path1)
            img1 = self.transform(img1)
            img2 = Image.open(path2)
            img2 = self.transform(img2)
            return img1, img2, val


if __name__ == "__main__":
    trainset = Full_Dataset("/home/anhnt596/Palm-Recognition/data/data", train=False)

    print(trainset[0][0].shape)
