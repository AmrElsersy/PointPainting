"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: KITTI & Cityscapes Visualization
"""
import argparse
import enum
import torch
from torch.utils.data import  DataLoader
import numpy as np 
import cv2
from dataset import KittiSemanticDataset
from utils.label import id2label, trainId2label
from utils.utils import tensor_to_cv2
import torch.utils.tensorboard as tensorboard

class KittiVisualizer:
    def __init__(self):
        self.scene_width = 1000
        self.scene_height = 600 # for horizontal visualization

    def add_semantic_to_image(self, image, semantic):
        return cv2.addWeighted(cv2.resize(image, (semantic.shape[1], semantic.shape[0]) )
                               .astype(np.uint8), 1, 
                               semantic.astype(np.uint8), .5, 
                               0, cv2.CV_32F)

    """
        Visualize image & predicted semantic label_ids & label semantic label_ids
        Args:
            image: input image of shape (342, 1247)
            semantic: output model semantic map of shape () 
    """
    def visualize_test(self, image, semantic, label):
        self.scene_width = 680
        self.__visualize(image, semantic, label)
                
    def visualize(self, image, semantic, label):
        semantic = self.semantic_to_color(semantic)
        label = self.semantic_to_color(label)
        self.__visualize(image, semantic, label)

    def visualize_horizontal(self, image, semantic):
        scene_height = self.scene_height        
        image_h, image_w = image.shape[:2]
        semantic_h, semantic_w = semantic.shape[:2]

        new_image_width = int(image_h * scene_height / image_h)
        new_semantic_width = int(semantic_h * scene_height / semantic_h)

        image = cv2.resize(image, (scene_height, new_image_width))
        semantic = cv2.resize(semantic, (scene_height, new_semantic_width)) 

        total_image = np.zeros((scene_height, new_image_width + new_semantic_width, 3), dtype=np.uint8)

        total_image[:, :new_image_width, :] = image
        total_image[:, new_image_width:, :] = semantic

        self.__show(total_image)

    def __visualize(self, image, semantic, label):    

        scene_width = self.scene_width        
        image_h, image_w = image.shape[:2]
        semantic_h, semantic_w = semantic.shape[:2]
        label_h, label_w = label.shape[:2]

        new_image_height = int(image_h * scene_width / image_w)
        new_semantic_height = int(semantic_h * scene_width / semantic_w)
        new_label_height = int(label_h * scene_width / label_w)

        image = cv2.resize(image, (scene_width, new_image_height))
        semantic = cv2.resize(semantic, (scene_width, new_semantic_height), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (scene_width, new_label_height))

        total_image = np.zeros((new_image_height + new_semantic_height + new_label_height, 
                                scene_width, 3), dtype=np.uint8)

        total_image[:new_image_height, :, :] = image
        total_image[new_image_height:new_image_height + new_label_height, :, :] = semantic
        total_image[new_image_height + new_label_height:, :, :] = label

        self.__show(total_image)

    def __show(self, image):
        cv2.namedWindow('total_image')
        def print_img(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(image[y,x])
        cv2.setMouseCallback('total_image', print_img)
        cv2.imshow("total_image", image)
        self.__show_2D()

    def semantic_to_color(self, semantic):
        r = np.zeros((semantic.shape[:2])).astype(np.uint8)
        g = np.zeros((semantic.shape[:2])).astype(np.uint8)
        b = np.zeros((semantic.shape[:2])).astype(np.uint8)

        for key in trainId2label:
            label = trainId2label[key]
            if key == 255 or key == -1:
                continue
            id = key
            color = label.color
            indices = semantic == id
            r[indices], g[indices], b[indices] = color

        semantic = np.stack([b, g, r], axis=2)
        return semantic

    def __show_2D(self):
        self.pressed_btn = cv2.waitKey(0) & 0xff
        
def main():
    dataset = KittiSemanticDataset(mode = 'semantic')
    visualizer = KittiVisualizer()
    for i in range(len(dataset)):
        image, semantic = dataset[i]
        print(image.shape)
        semantic = visualizer.semantic_to_color(semantic)
        new_img = visualizer.add_semantic_to_image(image, semantic)
        cv2.imshow('image', new_img)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', action='store_true', help='tensorboard visualization')
    parser.add_argument('--logdir', type=str, default='checkpoints/tensorboard', help='tensorboard log directory')
    parser.add_argument('--batch_size', type=int, default=50,help='num of images in each tensorboard batch vis')
    parser.add_argument('--stop', type=int, default=4, help='number of batches to be visualized in tensorboard')
    args = parser.parse_args()

    if not args.tensorboard:
        main()
    else:
        # Tensorboard
        dataset = KittiSemanticDataset(mode = 'color')
        visualizer = KittiVisualizer()
        writer = tensorboard.SummaryWriter(args.logdir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        batch = 0
        for images, semantics in dataloader:
            batch += 1
            images_colored = torch.zeros_like(images)
            for i, image in enumerate(images):
                semantic = semantics[i]
                image = image.numpy()
                semantic = semantic.numpy()
                image_colored = visualizer.add_semantic_to_image(image, semantic)
                image_colored = cv2.cvtColor(image_colored, cv2.COLOR_RGB2BGR)
                images_colored[i] = torch.from_numpy(image_colored)

            writer.add_images("images_colored", images_colored, global_step=batch, dataformats="NHWC")
            print ("*" * 60, f'\n\n\t Saved {args.batch_size} images with Step {batch}. run tensorboard @ project root')
            if batch == args.stop:
                writer.close()
                break

