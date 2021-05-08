"""

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
import torch.nn.functional as F

from dataset import KittiSemanticDataset, cityscapes_dataset
from visualization import KittiVisualizer
from model.DDRNet_23_slim import DDRNet_23_slim

def test(args):
    dataset = cityscapes_dataset()
    # dataset = KittiSemanticDataset()
    visualizer = KittiVisualizer()
    
    model = DDRNet_23_slim(pretrained=True, path = args.path)
    model.eval()

    for i in range(len(dataset)):
        image, semantic = dataset[i]
        # print(image.size, semantic.size)
        # (1, 3, 512, 1024) required

        # =============== Preprocessing =============== 
        # to numpy
        image = np.asarray(image)
        semantic = np.asarray(semantic)
        # resize
        # print(image.shape, semantic.shape)
        new_shape = (1024,512)
        image = cv2.resize(image, new_shape)
        semantic = cv2.resize(semantic, new_shape)
        # print(image.shape, semantic.shape)
        # to tensor
        image = transforms.ToTensor()(image)
        semantic = transforms.ToTensor()(semantic)
        # print(image.size(), semantic.size())

        image = torch.unsqueeze(image, 0)
        pred_semantic = model(image)
        
        size = semantic.size()[-2:]
        pred_semantic = F.interpolate(input=pred_semantic, size=size, mode='bilinear', align_corners=True)
        print(pred_semantic.shape)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='checkpoints/weights/DDRNet_23_slim.pth')
    args = parser.parse_args()
    test(args)