"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Utils functions used in many places 
"""

import os, time, enum
from PIL import Image
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.transforms as transforms
import torchvision
import numpy as np 
import cv2

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

def read_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def preprocessing_kitti(image):
    if type(image) != np.array:
        image = np.asarray(image)
    new_shape = (1024,512)
    image = cv2.resize(image, new_shape)
    image = transforms.ToTensor()(image)
    # image = torch.from_numpy(image.copy().transpose(2,0,1)).unsqueeze(0).to(device) / 255

    image = image.unsqueeze(0).to(device)
    return image

def preprocessing_cityscapes(image):
    mean=(0.3257, 0.3690, 0.3223)
    std=(0.2112, 0.2148, 0.2115)

    image = transforms.ToTensor()(image)
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)[:, None, None]
    std = torch.as_tensor(std, dtype=dtype, device=device)[:, None, None]
    image = image.sub_(mean).div_(std).clone()
    image = image.unsqueeze(0).to(device)
    return image

def postprocessing(pred):
    pred = pred.argmax(dim=1).squeeze().detach().cpu().numpy()
    return pred

def tensor_to_cv2(image):
    return image.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
