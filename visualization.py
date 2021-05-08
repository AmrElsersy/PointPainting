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
from utils.utils import read_image
from dataset import KittiSemanticDataset
from utils.label import Label, name2label, id2label

class KittiVisualizer:
    def __init__(self):
        self.scene_width = 1000

    def add_semantic_to_image(self, image, semantic):
        return cv2.addWeighted(image, 1, semantic, .6, 0)


    """
        Visualize image & predicted semantic label_ids & label semantic label_ids
        Args:
            image: input image of shape (342, 1247)
            semantic: output model semantic map of shape () 
    """
    def visualize(self, image, semantic, label):

        semantic = self.semantic_to_color(semantic)
        label = self.semantic_to_color(label)

        scene_width = self.scene_width        
        image_h, image_w = image.shape[:2]
        semantic_h, semantic_w = semantic.shape[:2]
        label_h, label_w = label.shape[:2]

        new_image_height = int(image_h * scene_width / image_w)
        new_semantic_height = int(semantic_h * scene_width / semantic_w)
        new_label_height = int(label_h * scene_width / label_w)

        image = cv2.resize(image, (scene_width, new_image_height))
        semantic = cv2.resize(semantic, (scene_width, new_semantic_height))
        label = cv2.resize(label, (scene_width, new_label_height))

        total_image = np.zeros((new_image_height + new_semantic_height + new_label_height, 
                                scene_width, 3), dtype=np.uint8)

        total_image[:new_image_height, :, :] = image
        total_image[new_image_height:new_image_height + new_label_height, :, :] = semantic
        total_image[new_image_height + new_label_height:, :, :] = label

        cv2.namedWindow('total_image')
        def print_img(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(total_image[y,x])
        cv2.setMouseCallback('total_image', print_img)
        cv2.imshow("total_image", total_image)
        self.__show_2D()
    
    def visualize_semantic_bev(self, image, semantic, pointcloud):
        image = self.add_semantic_to_image(image, semantic)
        bev = pointcloud_to_bev(pointcloud)

    def semantic_to_color(self, semantic):
        r = np.zeros((semantic.shape[:2])).astype(np.uint8)
        g = np.zeros((semantic.shape[:2])).astype(np.uint8)
        b = np.zeros((semantic.shape[:2])).astype(np.uint8)

        # print(id2label)
        for id in id2label:
            if id <= 0:
                continue
            color = id2label[id].color
            indices = (semantic == id)[:,:,0]
            r[indices], g[indices], b[indices] = color

        semantic = np.stack([b, g, r], axis=2)
        return semantic

    def __show_2D(self):
        self.pressed_btn = cv2.waitKey(0) & 0xff
        
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
