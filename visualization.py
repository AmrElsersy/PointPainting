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
        self.scene_width = 1000

    def add_semantic_to_image(self, image, semantic):
        return cv2.addWeighted(image, 1, semantic, .6, 0)

    def visualize(self, image, semantic, label):
        # all will have the same width, just map the height to the same ratio to have the same image
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

        cv2.imshow("total_image", total_image)
        self.__show_2D()
    
    def visualize_semantic_bev(self, image, semantic, pointcloud):
        image = self.add_semantic_to_image(image, semantic)
        bev = pointcloud_to_bev(pointcloud)

    def semantic_ids_to_color(self, semantic):    
        colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        colors = [[255,0,0]]
        r = np.zeros_like(semantic).astype(np.uint8)
        g = np.zeros_like(semantic).astype(np.uint8)
        b = np.zeros_like(semantic).astype(np.uint8)
        r[semantic == 1], g[semantic == 1], b[semantic == 1] = colors[random.randrange(0,len(colors))]
        coloured_semantic = np.stack([r, g, b], axis=2)
        return coloured_semantic

    def __show_2D(self):
        self.pressed_btn = cv2.waitKey(0) & 0xff
        
def main():
    dataset = KittiSemanticDataset(mode = 'semantic')
    visualizer = KittiVisualizer()
    for i in range(len(dataset)):
        image, semantic = dataset[i]
        visualizer.visualize(image, semantic, semantic)

        # new_img = visualizer.add_semantic_to_image(image, semantic)
        # cv2.imshow('image', new_img)
        # if cv2.waitKey(0) == 27:
        #     cv2.destroyAllWindows()
        #     break

if __name__ == '__main__':    
    main()
