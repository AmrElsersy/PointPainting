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
from pointpainting import PointPainter

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

    # Fusion
    painter = PointPainter()

    video = KittiVideo(
        video_root=args.video_path,
        calib_root=args.calib_path
    )

    visualizer = Visualizer(args.mode)

    frames = []
    if args.mode == '2d':
        frame_shape = (750, 900)
    else:
        frame_shape = (1280, 720)
    avg_time = 0

    for i in range(len(video)):
        image, pointcloud, calib = video[i]


        t1 = time_synchronized()
        input_image = preprocessing_kitti(image)
        semantic = bisenetv2(input_image)
        semantic = postprocessing(semantic)
        t2 = time_synchronized()

        painted_pointcloud = painter.paint(pointcloud, semantic, calib)
        t3 = time_synchronized()

    
        print(f'Time of bisenetv2 = {1000 * (t2-t1)} ms')
        print(f'Time of pointpainting = {1000 * (t3-t2)} ms')

        if args.mode == '3d':
            screenshot = visualizer.visuallize_pointcloud(painted_pointcloud, blocking=False)
            print(screenshot.shape)
            frames.append(screenshot)
        else:
            color_image = visualizer.get_colored_image(image, semantic)
            if args.mode == 'img':
                frames.append(color_image)
                cv2.imshow('color_image', color_image)
            elif args.mode == '2d':
                scene_2D = visualizer.get_scene_2D(color_image, painted_pointcloud, calib)
                frames.append(scene_2D)
                cv2.imshow('scene', scene_2D)       

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break

        # if i == 20:
        #     break

        avg_time += (time.time()-t1)
        print(f'{i} sample')

    # Time & FPS
    avg_time /= len(video)
    FPS = 1 / avg_time
    print("Average Time",round(avg_time*1000,2), "ms  FPS", round(FPS,2))
    # Save Video
    save_path = os.path.join(args.save_path, 'demo.mp4')
    out_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 18, frame_shape)
    for frame in frames:
        frame = cv2.resize(frame, frame_shape)
        out_video.write(frame)
    print(f'Saved Video @ {save_path}')


def boundary_3d_modify():
    from bev_utils import boundary
    boundary['minY'] = -30
    boundary['maxY'] = 30
    boundary['maxX'] = 100
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, 
    # default='Videos/2/2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync',)
    # default='Videos/1/2011_09_26/2011_09_26_drive_0048_sync',)
    default='Videos/1/2011_09_26_drive_0093_sync/2011_09_26')

    parser.add_argument('--calib_path', type=str, 
    # default='Videos/2/2011_09_26_calib/2011_09_26',)
    default='Videos/1/2011_09_26_calib/2011_09_26',)

    parser.add_argument('--weights_path', type=str, default='BiSeNetv2/checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--save_path', type=str, default='Videos',)
    parser.add_argument('--mode', type=str, default='3d', choices=['img', '2d', '3d'], 
    help='visualization mode .. img is semantic image .. 2d is semantic + bev .. 3d is colored pointcloud')
    args = parser.parse_args()
    if args.mode == '3d':
        boundary_3d_modify()

    main(args)



