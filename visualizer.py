import sys
import cv2
import numpy as np

sys.path.insert(0, 'BiSeNetv2')

from BiSeNetv2.visualization import KittiVisualizer
from bev_utils import pointcloud_to_bev

from mayavi import mlab


class Visualizer():
    def __init__(self):
        self.__semantic_visualizer = KittiVisualizer()
        self.scene_2D_width = 750
        self.user_press =None

    def visuallize_pointcloud(self, pointcloud):
        pointcloud = self.__to_numpy(pointcloud)
        mlab.points3d(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], 
                    colormap='gnuplot', scale_factor=1, mode="point",  figure=self.figure)
        mlab.show(stop=True)
        
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

