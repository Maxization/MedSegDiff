import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from glob import glob
import ntpath

class CBISDDSMDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training', plane = False):

        images_path = glob(os.path.join(data_path, '*.tif'))
        original_images_path = []

        for item in images_path:
            if '_mask' not in item:
                original_images_path.append(item)

        original_masks_path = []
        for item in original_images_path:
            original_masks_path.append(item[:-4] + '_mask.tif')

        self.name_list = original_images_path
        self.label_list = original_masks_path
        self.data_path = data_path
        self.mode = mode

        self.transform = transform
        tran_list = [transforms.Resize((320, 512)), transforms.ToTensor(), ]
        self.mask_transform = transforms.Compose(tran_list)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = name
        
        msk_path = self.label_list[index]

        img = Image.open(img_path).convert('L')
        mask = Image.open(msk_path).convert('L')

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.mask_transform(mask)

        return (img, mask, name)