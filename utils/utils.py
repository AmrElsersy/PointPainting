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
