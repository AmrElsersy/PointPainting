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


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def preprocessing_ddrnet(image):
    # to numpy
    if type(image) != np.array:
        image = np.asarray(image)
    # resize
    new_shape = (1024,512)
    image = cv2.resize(image, new_shape)
    # to tensor
    image = transforms.ToTensor()(image)
    return image
