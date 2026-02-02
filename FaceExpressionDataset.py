import PIL.Image
import torch
import torchvision
from skimage import io, transform
import pandas as pd
import PIL
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from typing import Union 


import warnings
warnings.filterwarnings("ignore")

plt.ion()

class FaceExpressionDataset(Dataset):
    def __init__(self, root_dir = "", transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.anger_len = len(os.listdir(f"{self.root_dir}/anger"))
        self.blink_len = len(os.listdir(f"{self.root_dir}/blink"))
        self.frown_len = len(os.listdir(f"{self.root_dir}/frown"))
        self.neutral_len = len(os.listdir(f"{self.root_dir}/neutral"))
        self.smile_len = len(os.listdir(f"{self.root_dir}/smile"))
    def __len__(self):
        sum = self.anger_len + self.blink_len + self.frown_len + self.neutral_len + self.smile_len-3
        return sum
    def __getitem__(self, ind:Union[int,torch.Tensor]):
        if torch.is_tensor(ind):
            ind = ind.tolist()
        filename:str
        label:int
        flag = False
        if ind < self.anger_len:
            filename = f"{self.root_dir}/anger/anger_{ind+1}.png"
            label = 0
            flag = True
        else:ind-=self.anger_len-1
        
        if ind < self.blink_len and not flag:
            filename = f"{self.root_dir}/blink/blink_{ind+1}.png"
            label = 1
            flag = True
        else:ind-=self.blink_len-1

        if ind < self.frown_len and not flag:
            filename = f"{self.root_dir}/frown/frown_{ind+1}.png"
            label = 2
            flag = True
        else:ind-=self.frown_len
        
        if ind < self.neutral_len and not flag:
            filename = f"{self.root_dir}/neutral/neutral_{ind+1}.png"
            label = 3
            flag = True
        else:ind-=self.neutral_len
        
        if ind < self.smile_len and not flag:
            filename = f"{self.root_dir}/smile/smile_{ind+1}.png"
            label = 4
            flag = True

        image = PIL.Image.open(filename)
        sample = {"image":image}

        if self.transform:
            image_tensor = self.transform(image)

        return image_tensor,label

class ToTensor(object):

    def __call__(self, sample):
        image = sample["image"]

        better = np.transpose(image,(2,0,1))
        return {"image":torch.from_numpy(better)}

set = FaceExpressionDataset("dataset", transform= transforms.Compose([torchvision.transforms.Resize((256,256)),
                                                                      torchvision.transforms.ToTensor()]))


# print(set[7385])