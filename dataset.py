"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: KITTI & Cityscapes Dataset modules to read images with semantic annotations
"""

import os, time, enum
from PIL import Image
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.transforms as transforms
import torchvision
import numpy as np 
import cv2
from utils.label import labels, id2label
from utils.utils import TransformationTrain

class KittiSemanticDataset(Dataset):
    def __init__(self, root = 'data/KITTI', split = 'train', mode = 'semantic', transform = None):
        self.transform = transform

        assert split in ['train', 'test']
        self.split = 'training' if split == 'train' else 'testing'

        self.root = os.path.join(root, self.split)

        assert mode in ['semantic', 'color']
        self.mode = mode

        # paths of images and labels 
        self.imagesPath = os.path.join(self.root, "image_2")
        self.semanticPath = os.path.join(self.root, "semantic")
        self.colorPath = os.path.join(self.root, "semantic_rgb")

        # list all images / labels paths
        images_names   = sorted(os.listdir(self.imagesPath))
        semantic_names = sorted(os.listdir(self.semanticPath))
        color_names    = sorted(os.listdir(self.colorPath))

        # add the root path to images names
        self.images_paths   = [os.path.join(self.imagesPath, name) for name in images_names]
        self.semantic_paths = [os.path.join(self.semanticPath, name) for name in semantic_names]
        self.color_paths    = [os.path.join(self.colorPath, name) for name in color_names]

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        semantic_path = self.semantic_paths[index]
        color_path = self.color_paths[index]

        image = self.read_image(image_path)
        semantic = None

        if self.mode == 'semantic':
            semantic = self.read_image(semantic_path)
        elif self.mode == 'color':
            semantic = self.read_image(color_path)

        image = np.asarray(image)
        semantic = np.asarray(semantic)

        # its 3 identical channels (each one is semantic map)
        if self.mode == 'semantic': 
            semantic = semantic[:,:,0]

        # print(semantic.shape)
        shape = (1024, 512)
        image = cv2.resize(image, shape)
        semantic = cv2.resize(semantic, shape, interpolation=cv2.INTER_NEAREST)

        if self.split == 'training':
            semantic = self.remove_ignore_index_labels(semantic)

        if self.transform:
            image = self.transform(image)
        return image, semantic

    def __len__(self):
        return len(self.images_paths)

    def read_image(self, path):
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def remove_ignore_index_labels(self, semantic):
        for id in id2label:
            label = id2label[id]
            trainId = label.trainId
            semantic[semantic == id] = trainId
        # print(np.unique(semantic))
        return semantic

def create_train_dataloader(root = 'data/KITTI', batch_size = 4):
    transform = transforms.ToTensor()
    # transform = TransformationTrain(scales=[0.25, 2.], cropsize=[512, 1024])
    dataset = KittiSemanticDataset(root = root, split='train', transform=transform)
    indices = list(range(0, 180))
    train_subset = Subset(dataset, indices)
    dataloader = DataLoader(train_subset, batch_size, shuffle=True)
    return dataloader

def create_val_dataloader(root = 'data/KITTI', batch_size = 1):
    transform = transforms.ToTensor()
    dataset = KittiSemanticDataset(root = root, split='train', transform=transform)
    indices = list(range(180, len(dataset)))
    val_subset = Subset(dataset, indices)
    dataloader = DataLoader(val_subset, batch_size, shuffle=False)
    return dataloader

def create_test_dataloader(root = 'data/KITTI', batch_size = 1):
    transform = transforms.ToTensor()
    dataset = KittiSemanticDataset(root = root, split='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader

def cityscapes_dataset(split = 'test', path = 'data/Cityscapes', mode ='semantic'):
    # types: 'color', 'semantic'
    dataset = torchvision.datasets.Cityscapes(path, split= split, mode='fine', target_type= mode)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['kitti', 'cityscapes'], default='kitti')
    parser.add_argument('--mode', type=str, choices=['semantic', 'color'], default='color')
    args = parser.parse_args()

    # ================= Prinint Values ===================
    def image_callback(event,x,y,flags,param):
        print(image[y,x])
    def semantic_callback(event,x,y,flags,param):
        print(semantic[y,x])

    cv2.namedWindow('image')
    cv2.namedWindow('semantic')
    cv2.setMouseCallback('image',image_callback)
    cv2.setMouseCallback('semantic',semantic_callback)
    # ====================================================

    if args.dataset == 'cityscapes':
        dataset = cityscapes_dataset(mode = args.mode)
    else:
        dataset = KittiSemanticDataset('data/KITTI', mode=args.mode)
    
    for i in range(len(dataset)):
        image, semantic = dataset[i]
        # print(image.size, semantic.size)
    
        if args.dataset == 'cityscapes':
            image = np.asarray(image)
            semantic = np.asarray(semantic)

        print(image.shape, semantic.shape)
        new_shape = (1024,512)
        image = cv2.resize(image, new_shape)
        semantic = cv2.resize(semantic, new_shape)
        print(image.shape, semantic.shape)

        cv2.imshow('image', image)
        cv2.imshow('semantic', semantic)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':    
    main()
