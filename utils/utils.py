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


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def preprocessing_ddrnet(image):
    # to numpy
    if type(image) != np.array:
        image = np.asarray(image)
    # resize
    new_shape = (1024,512)
    image = cv2.resize(image, new_shape)
    # to tensor
    image = transforms.ToTensor()(image)
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

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)
