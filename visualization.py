"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: KITTI & Cityscapes Visualization
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
from utils import read_image

class VisualizeMode(enum.Enum):
    WEIGHTED = 0
    SEPERATED = 1
    LIDAR = 2

class KittiVisualizer:
    def __init__(self):
        return

    def visualize(self):
        return


