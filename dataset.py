"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Cityscapes Dataset module to read images with semantic annotations
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

def kitti_semantic_dataset():
    return


def print_img_value(img, x,y):
    print(img[y,x])



def semantic_callback(event,x,y,flags,param):
    # if event == cv2.EVENT_LBUTTONDOWN:
        print_img_value(semantic, x, y)
def labelIds_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print_img_value(labelIds, x, y)
def instanceIds_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print_img_value(instanceIds, x, y)

cv2.namedWindow('image')
cv2.namedWindow('semantic')
cv2.setMouseCallback('image',semantic_callback)
cv2.setMouseCallback('semantic',semantic_callback)

# cv2.namedWindow('labelIds')
# cv2.setMouseCallback('labelIds',labelIds_callback)
# cv2.namedWindow('instanceIds')
# cv2.setMouseCallback('instanceIds',instanceIds_callback)

def cityscapes_dataset(mode = 'train', path = 'data/Cityscapes'):
    # dataset = torchvision.datasets.Cityscapes(path, split= mode, mode='fine', target_type=['instance', 'color', 'polygon'])
    dataset = torchvision.datasets.Cityscapes(path, split= mode, mode='fine', target_type= 'semantic')
    return dataset

if __name__ == '__main__':    
    dataset = cityscapes_dataset()
    for i in range(len(dataset)):
        image, semantic = dataset[i]
        # image, (inst, col, poly) = dataset[i]
        # print(inst, col, poly)

        image = np.asarray(image)
        semantic = np.asarray(semantic)
        print(type(image), type(semantic))
        print(image.shape, semantic.shape)

        new_shape = (1280,720)
        image = cv2.resize(image, new_shape)
        semantic = cv2.resize(semantic, new_shape)

        cv2.imshow('image', image)
        cv2.imshow('semantic', semantic)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break


def test():
    # all of them are shape (1024, 2048, 3)
    semantic = read_image('data/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png')
    instanceIds = read_image('data/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_instanceIds.png')
    labelIds = read_image('data/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png')
    semantic = read_image('data/1.png')

    new_shape = (1280,720)
    semantic = cv2.resize(semantic, new_shape)
    labelIds = cv2.resize(labelIds, new_shape)
    # instanceIds = cv2.resize(instanceIds, new_shape)

    cv2.imshow("semantic", semantic)
    cv2.imshow("labelIds", labelIds)
    # cv2.imshow("instanceIds", instanceIds)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
