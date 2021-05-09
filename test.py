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
from utils.utils import preprocessing_ddrnet

names = ("background", "floor", "bed", "cabinet,wardrobe,bookcase,shelf",
        "person", "door", "table,desk,coffee", "chair,armchair,sofa,bench,swivel,stool",
        "rug", "railing", "column", "refrigerator", "stairs,stairway,step", "escalator", "wall",
        "dog", "plant")

colors  = np.array([[0, 0, 0],
                    [0, 0, 255],
                    [0, 255, 0],
                    [0, 255, 255],
                    [255, 0, 0 ],
                    [255, 0, 255 ], 
                    [255, 255, 0 ],
                    [255, 255, 255 ],
                    [0, 0, 128 ],
                    [0, 128, 0 ],
                    [128, 0, 0 ],
                    [0, 128, 128 ],
                    [128, 0, 0 ],
                    [128, 0, 128 ],
                    [128, 128, 0 ],
                    [128, 128, 128 ],
                    [192, 192, 192 ],
                    [192, 192, 192 ],
                    [192, 192, 192 ]], dtype=np.uint8)

def colorEncode(labelmap, mode='BGR'):
    global colors
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] *  np.tile(colors[label],(labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def test(args):
    # dataset = cityscapes_dataset()
    dataset = KittiSemanticDataset()
    visualizer = KittiVisualizer()
    
    model = DDRNet_23_slim(pretrained=True, path = args.path)
    model.eval()

    for i in range(len(dataset)):
        image, semantic = dataset[i]
        # print(image.size, semantic.size)
        # (1, 3, 512, 1024) required

        image = preprocessing_ddrnet(image)
        semantic = preprocessing_ddrnet(semantic)
        # print(image.size(), semantic.size())
        image = torch.unsqueeze(image, 0)
        size = image.size()[-2:]

        # print(image)
        pred_semantic = model(image)
        
        pred_semantic = F.interpolate(input=pred_semantic, size=size, mode='bilinear', align_corners=True)

        # get image back
        image = image.squeeze().detach().numpy().transpose(1,2,0)
        
        cv2.imshow('image', image)
        # names = ['road', 'pedestrian', 'car']
        # for i, id in enumerate([2, 18, 10]):
        #     pred = pred_semantic[0][id].detach().numpy().astype(np.uint8)
        #     cv2.imshow(names[i], pred)
        #     print(pred, end='\n')

        print(pred_semantic.shape)
        pred_semantic = torch.argmax(pred_semantic, dim=1).squeeze(0).cpu().numpy()
        print(pred_semantic)
        # return
        print(pred_semantic.shape)
        pred_semantic_color = colorEncode(pred_semantic)
        cv2.imshow('pred', pred_semantic_color)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='checkpoints/weights/DDRNet_23_slim.pth')
    args = parser.parse_args()
    test(args)