import os
import cv2
import numpy as np
import argparse
import time
import torch

from KittiCalibration import KittiCalibration
from visualizer import Visualizer
from BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from BiSeNetv2.utils.utils import preprocessing_kitti, postprocessing
from pointpainting import PointPainter
from bev_utils import boundary

import tensorrt as trt

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main(args):

    # Semantic Segmentation
    bisenetv2 = BiSeNetV2()
    checkpoint = torch.load(args.weights_path, map_location=dev)
    bisenetv2.load_state_dict(checkpoint['bisenetv2'], strict=False)
    bisenetv2.eval()
    bisenetv2.to(device)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    print(TRT_LOGGER, builder)
    
    # Fusion
    painter = PointPainter()

    # Visualizer
    visualizer = Visualizer(args.mode)

    image = cv2.imread(args.image_path)
    pointcloud = np.fromfile(args.pointcloud_path, dtype=np.float32).reshape((-1, 4))

    # if calib file is in kitti video format
    # calib = KittiCalibration(args.calib_path, from_video=True)
    # if calib file is in normal kitti format
    calib = KittiCalibration(args.calib_path)


    t1 = time_synchronized()
    input_image = preprocessing_kitti(image)
    print(f'Time of preprocessing = {1000 * (time_synchronized()-t1)} ms')
    print(input_image.shape)
    semantic = bisenetv2(input_image)
    t2 = time_synchronized()
    semantic = postprocessing(semantic)
    t3 = time_synchronized()

    painted_pointcloud = painter.paint(pointcloud, semantic, calib)
    t4 = time_synchronized()

    print(f'Time of bisenetv2 = {1000 * (t2-t1)} ms')
    print(f'Time of postprocesssing = {1000 * (t3-t2)} ms')
    print(f'Time of pointpainting = {1000 * (t4-t3)} ms')

    if args.mode == '3d':
        visualizer.visuallize_pointcloud(painted_pointcloud, blocking=True)
    else:
        color_image = visualizer.get_colored_image(image, semantic)
        scene_2D = visualizer.get_scene_2D(color_image, painted_pointcloud, calib)
        scene_2D = cv2.resize(scene_2D, (600, 900))
        cv2.imshow("scene", scene_2D)

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='Kitti_sample/image_2/000038.png')
    parser.add_argument('--pointcloud_path', type=str, default='Kitti_sample/velodyne/000038.bin')
    parser.add_argument('--calib_path', type=str, default='Kitti_sample/calib/000038.txt')
    parser.add_argument('--weights_path', type=str, default='BiSeNetv2/checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--save_path', type=str, default='results',)
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'],
    help='visualization mode .. img is semantic image .. 2d is semantic + bev .. 3d is colored pointcloud')

    args = parser.parse_args()
    main(args)
    args.image_path = 'Kitti_sample/image_2/000031.png'
    main(args)



