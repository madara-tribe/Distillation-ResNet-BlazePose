import os
import sys
sys.path.append("../")
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import cv2
from models.teacher.blazebase import resize_pad, denormalize_detections

def call_data_loader(dst, bs, shuffle=True, num_worker=0):
    loader = data.DataLoader(
        dst, batch_size=bs, shuffle=True, num_workers=num_worker)
    return loader

class data_loader(data.Dataset):
    def __init__(self,
                 image_dir,
                 width,
                 height,
                 transform=None):
        self.width = width
        self.height = height
        train_img_dir = os.listdir(image_dir)
        train_img_dir.sort()
        # jpg image
        self.images = [os.path.join(image_dir, path) for path in train_img_dir]
        self.transforms = transform
    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = np.ascontiguousarray(image[:,::-1,::-1])
        _, img2, scale, pad = resize_pad(image) 
        image = Image.fromarray(img2)
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.images)


