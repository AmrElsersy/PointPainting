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
from utils.utils import preprocessing_cityscapes, preprocessing_kitti, colorEncode

from model.BiseNetv2 import BiSeNetV2
from config import cfg

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

# for colors generation
np.random.seed(123)

def test(args):
    if args.dataset == 'kitti':
        dataset = KittiSemanticDataset()
    else:
        dataset = cityscapes_dataset()

    visualizer = KittiVisualizer()
    
    # define model
    model = BiSeNetV2(19)
    checkpoint = torch.load(args.weight_path, map_location=dev)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)

    for i in range(len(dataset)):
        image, semantic = dataset[i]
        original = np.asarray(image.copy())

        # (1, 3, 512, 1024) required
        if args.dataset == 'kitti':
            image = preprocessing_kitti(image)
        else:
            image = preprocessing_cityscapes(image)

        print(image.shape)
        pred = model(image)
        pred = pred.argmax(dim=1).squeeze().detach().cpu().numpy()
        print(pred.shape)        

        # coloring
        palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        pred = palette[pred]
        print(pred.shape)

        # get numpy image back
        image = image.squeeze().detach().numpy().transpose(1,2,0)

        # save
        new_shape = (1024, 512)
        image = cv2.resize(image, new_shape)
        pred = cv2.resize(pred, new_shape)
        original = cv2.resize(original, new_shape)
        total = visualizer.add_semantic_to_image(original, pred)

        cv2.imshow('image',image)
        cv2.imshow('pred', pred)
        cv2.imshow('total', total)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break
        cv2.imwrite('./res.jpg', pred)

        # names = ['road', 'pedestrian', 'car']
        # for i, id in enumerate([2, 18, 10]):
        #     pred = pred_semantic[0][id].detach().numpy().astype(np.uint8)
        #     cv2.imshow(names[i], pred)
        #     print(pred, end='\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='checkpoints/BiseNetv2.pth',)
    parser.add_argument('--dataset', choices=['cityscapes', 'kitti'], default='kitti')
    args = parser.parse_args()
    test(args)
