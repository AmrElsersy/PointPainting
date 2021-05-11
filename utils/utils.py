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

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

def read_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def preprocessing_kitti(image):
    if type(image) != np.array:
        image = np.asarray(image)
    new_shape = (1024,512)
    image = cv2.resize(image, new_shape)
    image = transforms.ToTensor()(image)
    # image = torch.from_numpy(image.copy().transpose(2,0,1)).unsqueeze(0).to(device) / 255

    image = image.unsqueeze(0).to(device)
    return image

def preprocessing_cityscapes(image):
    mean=(0.3257, 0.3690, 0.3223)
    std=(0.2112, 0.2148, 0.2115)

    image = transforms.ToTensor()(image)
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)[:, None, None]
    std = torch.as_tensor(std, dtype=dtype, device=device)[:, None, None]
    image = image.sub_(mean).div_(std).clone()
    image = image.unsqueeze(0).to(device)
    return image



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

