from typing import Any, List, Tuple

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re

class COEP_Dataset(Dataset):
    def __init__(self, root_dir:str) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            #transforms.Resize((256,256)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.path = []
        for image in os.listdir(root_dir):
            self.path.append(os.path.join(root_dir, image))

    def __len__(self):
        return(len(self.path))
    
    def __getitem__(self, index):
        image_path = self.path[index]
        image_name = os.path.basename(image_path)

        img = Image.open(image_path)
        img = self.transform(img)

        personID = re.search(r'_(.*?)\(', image_name)
        personID = int(personID.group(1))-1

        res = []
        res.append(img)
        res.append(personID)
        return res

if __name__ == "__main__":
    _ = COEP_Dataset("/home/anhnt596/PalmPrint/database")

