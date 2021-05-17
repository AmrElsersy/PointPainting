import os
import cv2
import numpy as np
import argparse
import time
import torch

from KittiCalibration import KittiCalibration
from KittiVideo import KittiVideo
from visualizer import Visualizer
from BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from BiSeNetv2.utils.utils import preprocessing_kitti, postprocessing

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

def main(args):
    # Semantic Segmentation
    bisenetv2 = BiSeNetV2()
    checkpoint = torch.load(args.weights_path, map_location=dev)
    bisenetv2.load_state_dict(checkpoint['bisenetv2'], strict=False)
    bisenetv2.eval()
    bisenetv2.to(device)


    video = KittiVideo(
        video_root=args.video_path,
        calib_root=args.calib_path
    )

    visualizer = Visualizer()

    frames = []
    frame_shape = (750, 900)
    avg_time = 0

    for i in range(len(video)):
        t1 = time.time()
        image, pointcloud, calib = video[i]
        # print(image.shape, pointcloud.shape)

        input_image = preprocessing_kitti(image)
        semantic = bisenetv2(input_image)
        semantic = postprocessing(semantic)
        print(semantic.shape)

        color_image = visualizer.get_colored_image(image, semantic)
        scene_2D = visualizer.get_scene_2D(image, pointcloud, calib)
        # cv2.imshow('im', color_image)

        if args.mode == 'image':
            frames.append(color_image)
        elif args.mode == '2d':
            frames.append(scene_2D)
        elif args.mode == '3d':
            visualizer.visuallize_pointcloud(pointcloud)
            
        # if cv2.waitKey(0) == 27:
        #     cv2.destroyAllWindows()
        #     break
        avg_time += (time.time()-t1)
        print(f'{i} sample')
        if i == 10:
            break

    # Time & FPS
    avg_time /= len(video)
    FPS = 1 / avg_time
    print("Average Time",round(avg_time*1000,2), "ms  FPS", round(FPS,2))
    # Save Video
    save_path = os.path.join(args.save_path, 'demo.mp4')
    out_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, frame_shape)
    for frame in frames:
        frame = cv2.resize(frame, frame_shape)
        out_video.write(frame)
    print(f'Saved Video @ {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, 
    default='Videos/2/2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync',)
    parser.add_argument('--calib_path', type=str, 
    default='Videos/2/2011_09_26_calib/2011_09_26',)
    parser.add_argument('--weights_path', type=str, default='BiSeNetv2/checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--save_path', type=str, default='Videos',)
    parser.add_argument('--mode', type=str, default='2d', choices=['img', '2d', '3d'], 
    help='visualization mode .. img is semantic image .. 2d is semantic + bev .. 3d is colored pointcloud')
    args = parser.parse_args()
    main(args)



