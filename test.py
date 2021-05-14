"""

"""
from PIL import Image
import argparse
import torch
import torchvision.transforms.transforms as transforms
import numpy as np 
import cv2
import torch.nn.functional as F

from dataset import KittiSemanticDataset, cityscapes_dataset
from visualization import KittiVisualizer
from utils.utils import preprocessing_cityscapes, preprocessing_kitti, postprocessing

from model.BiseNetv2 import BiSeNetV2
from config import cfg

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

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
        x = semantic[:,:,1]- semantic[:,:,0]
        print(np.where(x != 0))
        # print(semantic.shape)
        # cv2.imshow('s', semantic)
        # cv2.waitKey(0)


        original = np.asarray(image.copy())

        # (1, 3, 512, 1024)
        if args.dataset == 'kitti':
            image = preprocessing_kitti(image)
        else:
            image = preprocessing_cityscapes(image)

        pred = model(image) # (19, 1024, 2048)
        pred = postprocessing(pred) # (1024, 2048) 

        # coloring
        pred = visualizer.semantic_to_color(pred) # (1024, 2048, 3)

        # get numpy image back
        image = image.squeeze().detach().numpy().transpose(1,2,0)

        # save
        pred_color = visualizer.add_semantic_to_image(original, pred)

        visualizer.visualize_test(original, pred, pred_color)
        if visualizer.pressed_btn == 27:
            cv2.destroyAllWindows()
            cv2.imwrite('./res.jpg', pred)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='checkpoints/BiseNetv2.pth',)
    parser.add_argument('--dataset', choices=['cityscapes', 'kitti'], default='kitti')
    args = parser.parse_args()
    test(args)
