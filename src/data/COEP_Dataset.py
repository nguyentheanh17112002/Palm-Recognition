from typing import Any, List, Tuple

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re
import random

class COEP_Dataset(Dataset):
    def __init__(self, root_dir:str, train: bool, random_seed: int = 42, test_size:int  = 10000) -> None:
        super().__init__()
        self.test_size = test_size
        self.random_seed = random_seed
        self.train = train
        self.root_dir = root_dir
        self.train_dir = os.path.join(self.root_dir,'train')
        self.test_dir = os.path.join(self.root_dir,'test')

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_path = []
        self.test_path = []
        valid_extensions = ['.jpg', '.jpeg', '.JPG', '.png', '.PNG', '.JPEG']

        for image in os.listdir(self.train_dir):
            if any(image.endswith(ext) for ext in valid_extensions):
                self.train_path.append(os.path.join(self.train_dir, image))

        for image in os.listdir(self.test_dir):
            if any(image.endswith(ext) for ext in valid_extensions):
                self.test_path.append(os.path.join(self.test_dir, image))
        
        self.test_pair = self._create_pair()

        


    def _create_pair(self):
        random.seed(self.random_seed)
        #create dictionary: {id : [img, img, img,...]}
        id_to_images = {}
        for path in self.test_path:
            image_name = os.path.basename(path)
            personID = re.search(r'_(.*?)\(', image_name)
            personID = int(personID.group(1))-1
            if personID not in id_to_images:
                id_to_images[personID] = []
            id_to_images[personID].append(path)

        same_person_pairs = []
        for _, paths in id_to_images.items():
            if len(paths) >= 8:
                same_person_pairs.extend([(paths[i], paths[j], 1) for i in range(7) for j in range(i + 1, 8)])
        
        different_person_pairs = []
        for _ in range(self.test_size - len(same_person_pairs)):
            person_ids = random.sample(list(id_to_images.keys()), 2)
            path1 = random.choice(id_to_images[person_ids[0]])
            path2 = random.choice(id_to_images[person_ids[1]])
            different_person_pairs.append((path1, path2, 0))

        res = same_person_pairs+different_person_pairs

        random.shuffle(res)

        return res


    def __len__(self):
        if self.train:
            return(len(self.train_path))
        else:
            return len(self.test_pair)
    
    def __getitem__(self, index):
        if self.train:
            '''
            Trainset chi co 96 nguoi :)) 1 nguoi co 9 anh
            thieu nguoi 6, 16, 52, 86
            '''
            minus = 1

            image_path = self.train_path[index]
            image_name = os.path.basename(image_path)

            img = Image.open(image_path)
            img = self.transform(img)

            personID = re.search(r'_(.*?)\(', image_name)
            personID = int(personID.group(1))

            if(personID >= 7 and personID < 17):
                minus = 2
            elif(personID >= 17 and personID < 53):
                minus = 3
            elif(personID >= 53 and personID < 87):
                minus = 4
            elif(personID >=87):
                minus = 5

            personID = personID - minus

            res = []
            res.append(img)
            res.append(personID)
            return res
        else:
            img1, img2, val = self.test_pair[index]

            img1 = Image.open(img1)
            img1 = self.transform(img1)
            img2 = Image.open(img2)
            img2 = self.transform(img2)

            return img1, img2, val



if __name__ == "__main__":
    testset = COEP_Dataset("/home/anhnt596/Palm-Recognition/data", train=False)
    print(trainset[-1])
