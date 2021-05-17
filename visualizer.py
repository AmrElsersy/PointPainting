import sys, time
import cv2
import numpy as np

sys.path.insert(0, 'BiSeNetv2')

from BiSeNetv2.visualization import KittiVisualizer
from bev_utils import pointcloud_to_bev

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

        # 3D
        self.__visualizer = o3d.visualization.Visualizer()
        self.__visualizer.create_window(width = 1280, height=720)
        self.__pcd = o3d.geometry.PointCloud()
        self.__visualizer.add_geometry(self.__pcd)

        self.__view_control = self.__visualizer.get_view_control()
        self.__view_control.set_zoom(300)
        self.__view_control.translate(100,0)        
        
        self.R = rotx(-np.pi/3) @ rotz(np.pi/2)

    def visuallize_pointcloud(self, pointcloud):
        pointcloud = pointcloud[:,:3]
        colors = np.zeros((pointcloud.shape[0], 3))

        self.__pcd.points = o3d.utility.Vector3dVector(pointcloud)
        self.__pcd.colors = o3d.utility.Vector3dVector(colors)

        self.__pcd.rotate(self.R, self.__pcd.get_center())

        # non blocking visualization
        self.__visualizer.add_geometry(self.__pcd)

        # control the view camera (must be after add_geometry())
        self.__view_control.translate(40,0)
        self.__view_control.set_zoom(0.1)

        self.__visualizer.update_renderer()
        self.__visualizer.poll_events()
        
        # save screenshot
        # self.__visualizer.capture_screen_image(path)

    def close_3d(self):
        self.__visualizer.destroy_window()

    def get_scene_2D(self, image, pointcloud, calib=None, visualize=False):
        bev = pointcloud_to_bev(pointcloud)
        print(bev.shape)
        scene_width = self.scene_2D_width        
        image_h, image_w = image.shape[:2]
        bev_h, bev_w = bev.shape[:2]

        # print(_image.shape, _bev.shape)

        new_image_height = int(image_h * scene_width / image_w)
        new_bev_height = int(bev_h * scene_width / bev_w)

        bev   = cv2.resize(bev,   (scene_width, new_bev_height) )
        image = cv2.resize(image, (scene_width, new_image_height) )

        image_and_bev = np.zeros((new_image_height + new_bev_height, scene_width, 3), dtype=np.uint8)

        image_and_bev[:new_image_height, :, :] = image
        image_and_bev[new_image_height:, :, :] = bev

        if visualize:
            cv2.imshow("scene 2D", image_and_bev)
        return image_and_bev

    def get_colored_image(self, image, semantic, visualize=False):
        semantic = self.__semantic_visualizer.semantic_to_color(semantic)
        color_image = self.__semantic_visualizer.add_semantic_to_image(image, semantic)
        if visualize:
            cv2.imshow('color_image', color_image)
        return color_image

    def __bev_to_colored_bev(self, bev):
    
        bev = (bev * 255).astype(np.uint8)

        # minZ = BEVutils.boundary["minZ"]
        # maxZ = BEVutils.boundary["maxZ"]
        # height_map = 255 - 255 * (height_map - minZ) / (maxZ - minZ) 
        # bev = np.dstack((intensity_map, height_map, density_map))

        return bev

