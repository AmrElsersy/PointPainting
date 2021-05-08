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
from dataset import KittiSemanticDataset

class KittiVisualizer:
    def __init__(self):
        return

    def add_semantic_to_image(self, image, semantic):
        return cv2.addWeighted(image, 1, semantic, .6, 0)

    def visualize(self, image, semantic, semantic_label):
        return

    def visualize_semantic_bev(self):
        return

    def semantic_ids_to_color(self, semantic):    
        colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        colors = [[255,0,0]]
        r = np.zeros_like(semantic).astype(np.uint8)
        g = np.zeros_like(semantic).astype(np.uint8)
        b = np.zeros_like(semantic).astype(np.uint8)
        r[semantic == 1], g[semantic == 1], b[semantic == 1] = colors[random.randrange(0,len(colors))]
        coloured_semantic = np.stack([r, g, b], axis=2)
        return coloured_semantic

def main():
    dataset = KittiSemanticDataset(mode = 'color')
    visualizer = KittiVisualizer()
    for i in range(len(dataset)):
        image, semantic = dataset[i]
        new_img = visualizer.add_semantic_to_image(image, semantic)
        cv2.imshow('image', new_img)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break
        
if __name__ == '__main__':    
    main()
