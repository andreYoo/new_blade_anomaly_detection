from torch.utils.data import Subset
from PIL import Image
from base.torchvision_dataset import TorchvisionDataset
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision.transforms as transforms
import random


class MyData(Dataset):

    def __init__(self,path,transforms):
        self.file = h5py.File(path,"r")
        self.data = np.array(self.file.get("samples"))
        self.transform = transforms

    def __len__(self):
        _shape = np.shape(self.data)
        return _shape[0]


    def __getitem__(self, index):
        img = self.data[index]
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = torch.FloatTensor(img)
        img = torch.unsqueeze(img,0)


        return img, index
