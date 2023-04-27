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
from model.ohem_loss import OhemCELoss
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
    model.load_state_dict(checkpoint['bisenetv2'], strict=False)
    model.eval()
    model.to(device)
    Loss = OhemCELoss(0.7)


    for i in range(len(dataset)):
        image, semantic = dataset[i]
        # cv2.imshow('s', semantic)
        # cv2.waitKey(0)

        original = np.asarray(image.copy())

        # (1, 3, 512, 1024)
        if args.dataset == 'kitti':
            image = preprocessing_kitti(image)
        else:
            image = preprocessing_cityscapes(image)

        # print(image)
        pred = model(image) # (19, 1024, 2048)

        # loss from just logits (not including aux)
        loss = Loss(pred, torch.from_numpy(semantic).unsqueeze(0))
        # print('loss',loss.item())

        pred = postprocessing(pred) # (1024, 2048) 

        # coloring
        # pred = np.stack([pred,pred,pred], axis=2).astype(np.uint8) # gray semantic
        pred = visualizer.semantic_to_color(pred) # (1024, 2048, 3)

        # get numpy image back
        image = image.squeeze().detach().numpy().transpose(1,2,0)

        # save
        # semantic = np.stack([semantic,semantic,semantic], axis=2)
        semantic = visualizer.semantic_to_color(semantic) # (1024, 2048, 3)

        # pred_color = visualizer.add_semantic_to_image(original, pred)
        visualizer.visualize_test(original, pred, semantic)
        # cv2.imshow('image', image)
        # cv2.imshow('pred', pred)
        # if cv2.waitKey(0) == 27:
        #     break
        if visualizer.pressed_btn == 27:
            cv2.destroyAllWindows()
            cv2.imwrite('./res.jpg', pred)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--dataset', choices=['cityscapes', 'kitti'], default='kitti')
    args = parser.parse_args()
    test(args)
