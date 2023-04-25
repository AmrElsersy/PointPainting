import argparse
import sys, time
import cv2
import numpy as np

sys.path.insert(0, 'BiSeNetv2')

from point_painting.BiSeNetv2.visualization import KittiVisualizer
from point_painting.bev_utils import pointcloud_to_bev
from point_painting.BiSeNetv2.utils.label import trainId2label

import open3d as o3d


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [c,-s, 0], 
        [s, c, 0], 
        [0, 0, 1]])

class Visualizer():
    def __init__(self):
        self.__semantic_visualizer = KittiVisualizer()
        self.scene_2D_width = 750
        self.user_press =None
        
        self.__visualizer = o3d.visualization.Visualizer()
        self.__visualizer.create_window(width = 1280, height=720)
        self.__pcd = o3d.geometry.PointCloud()
        self.__visualizer.add_geometry(self.__pcd)

        opt = self.__visualizer.get_render_option()   
        opt.background_color = np.asarray([0, 0, 0])
        self.zoom = 0.3 # smaller is zoomer

        self.__view_control = self.__visualizer.get_view_control()
        self.__view_control.translate(30,0)

        self.R = rotx(-np.pi/2.5) @ rotz(np.pi/2)

    def visuallize_pointcloud(self, pointcloud, blocking = False):

        semantics  = pointcloud[:,3]
        pointcloud = pointcloud[:,:3]
        colors = self.__semantics_to_colors(semantics)

        self.__pcd.points = o3d.utility.Vector3dVector(pointcloud)
        self.__pcd.colors = o3d.utility.Vector3dVector(colors)

        self.__pcd.rotate(self.R, self.__pcd.get_center())

        if blocking:
            o3d.visualization.draw_geometries([self.__pcd])
        else:
            # non blocking visualization
            self.__visualizer.add_geometry(self.__pcd)

            # control the view camera (must be after add_geometry())
            # self.__view_control.translate(30,0)
            self.__view_control.set_zoom(self.zoom)

            self.__visualizer.update_renderer()
            self.__visualizer.poll_events()
        
        screenshot = self.__visualizer.capture_screen_float_buffer()
        return (np.array(screenshot)*255).astype(np.uint8)[:,:,::-1]

    def __semantics_to_colors(self, semantics):
        # default color is black to hide outscreen points
        colors = np.zeros((semantics.shape[0], 3))

        for id in trainId2label:
            label = trainId2label[id]
            if id == 255 or id == -1:
                continue

            color = label.color
            indices = semantics == id
            colors[indices] = (color[0]/255, color[1]/255, color[2]/255)

        return colors

    def close_3d(self):
        self.__visualizer.destroy_window()

    def get_scene_2D(self, image, pointcloud, calib=None, visualize=False):
        bev = pointcloud_to_bev(pointcloud) # (600,600,4)
        bev = self.__bev_to_colored_bev(bev)

        scene_width = self.scene_2D_width        
        image_h, image_w = image.shape[:2]
        bev_h, bev_w = bev.shape[:2]

        new_image_height = int(image_h * scene_width / image_w)
        new_bev_height = int(bev_h * scene_width / bev_w)

        bev   = cv2.resize(bev,   (scene_width, new_bev_height), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (scene_width, new_image_height))

        image_and_bev = np.zeros((new_image_height + new_bev_height, scene_width, 3), dtype=np.uint8)

        image_and_bev[:new_image_height, :, :] = image
        image_and_bev[new_image_height:, :, :] = bev

        cv2.namedWindow('scene')
        def print_img(event,x,y,flags,param):
            # if event == cv2.EVENT_LBUTTONDOWN:
                print(image_and_bev[y,x])
        cv2.setMouseCallback('scene', print_img)
        if visualize:
            cv2.imshow("scene", image_and_bev)
        return image_and_bev

    def get_colored_image(self, image, semantic, visualize=False):
        semantic = self.__semantic_visualizer.semantic_to_color(semantic)
        color_image = self.__semantic_visualizer.add_semantic_to_image(image, semantic)
        if visualize:
            cv2.imshow('color_image', color_image)
        return color_image

    def __bev_to_colored_bev(self, bev):
        semantic_map = bev[:,:,3]
        shape = semantic_map.shape[:2]
        color_map = np.zeros((shape[0], shape[1], 3))

        for id in trainId2label:
            label = trainId2label[id]
            if id == 255 or id == -1:
                continue

            color = label.color
            color_map[semantic_map == id] = color[2], color[1], color[0]

        return color_map

    def visualize_painted_pointcloud(self, pointcloud):
        bev = pointcloud_to_bev(pointcloud) # (600,600,4)
        bev = self.__bev_to_colored_bev(bev) / 255.0
        return bev        


if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='2d', help='mode of visualization can be 2d or 3d')
    args = parser.parse_args()

    visualizer = Visualizer(mode=args.mode)

    path = 'tensorrt_inference/data/results_pointclouds'
    paths = sorted(os.listdir(path))
    pointclouds_paths = [os.path.join(path, pointcloud_path) for pointcloud_path in paths]

    for pointcloud_path in pointclouds_paths:
        pointcloud = np.fromfile(pointcloud_path, dtype=np.float32).reshape((-1, 4))
        bev = visualizer.visualize_painted_pointcloud(pointcloud=pointcloud)

        semantic_channel = pointcloud[:,3]
        semantic_channel = semantic_channel[semantic_channel != 255]
        semantic_df = pd.DataFrame(semantic_channel)
        # semantic_df.hist(bins=100)

        if args.mode == '2d':
            cv2.imshow("bev", bev)
            plt.show()
            if cv2.waitKey(0) == 27:
                exit()
        else:
            visualizer.visuallize_pointcloud(pointcloud, True)
