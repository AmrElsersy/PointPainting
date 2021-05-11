import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

from utils.utils import ToTensor
from model.BiseNetv2 import BiSeNetV2
from config import cfg

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--weight_path', type=str, default='checkpoints/BiseNetv2.pth',)
parse.add_argument('--path', dest='img_path', type=str, 
default='data/Cityscapes/leftImg8bit/test/berlin/berlin_000001_000019_leftImg8bit.png',)
args = parse.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

# define model
net = BiSeNetV2(19)
checkpoint = torch.load(args.weight_path, map_location=dev)
net.load_state_dict(checkpoint, strict=False)
net.eval()
net.to(device)

# prepare data
to_tensor = ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
image = cv2.imread(args.img_path)
original = np.copy(image)
# im = cv2.resize(im, (1024, 512))
image = to_tensor(dict(im=image, lb=None))['im'].unsqueeze(0).to(device)
# im = torch.from_numpy(im.copy().transpose(2,0,1)).unsqueeze(0).to(device) / 255
print(image.shape)

# inference
out = net(image)
out = out.argmax(dim=1).squeeze().detach().cpu().numpy()
print('output.shape', out.shape)

# coloring
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
pred = palette[out]

# to image numpy
image = image.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
# print(im.shape)

# save
new_shape = (1024, 512)
image = cv2.resize(image, new_shape)
pred = cv2.resize(pred, new_shape)
original = cv2.resize(original, new_shape)
total = cv2.addWeighted(original, 0.7, pred, 0.3, 0, dtype = cv2.CV_32F).astype(np.uint8)
cv2.imshow('image',image)
cv2.imshow('pred', pred)
cv2.imshow('total', total)
cv2.waitKey(0)
cv2.imwrite('./res.jpg', pred)
