"""

"""

import sys
from visualization import KittiVisualizer
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

from utils.utils import preprocessing_cityscapes, preprocessing_kitti, postprocessing, read_image, tensor_to_cv2
from model.BiseNetv2 import BiSeNetV2

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', type=str, default='checkpoints/BiseNetv2_150.pth',)
parser.add_argument('--path', dest='img_path', type=str, 
# default='data/Cityscapes/leftImg8bit/test/berlin/berlin_000001_000019_leftImg8bit.png',)
default='data/KITTI/testing/image_2/000169_10.png',)
args = parser.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

visualizer = KittiVisualizer()

# define model
model = BiSeNetV2(19)
checkpoint = torch.load(args.weight_path, map_location=dev)
model.load_state_dict(checkpoint['bisenetv2'], strict=False)
model.eval()
model.to(device)

image = read_image(args.img_path)
original = np.copy(image)

image = preprocessing_kitti(image)
# image = preprocessing_cityscapes(image)

# inference
pred = model(image)
pred = postprocessing(pred)


# coloring
pred = visualizer.semantic_to_color(pred)
# pred = np.stack([pred,pred,pred], axis=2).astype(np.uint8)

# visualize & save
total = visualizer.add_semantic_to_image(original, pred)
visualizer.visualize_test(original, pred, total)
# visualizer.visualize_horizontal(original, pred)
cv2.imwrite('./res.jpg', pred)
