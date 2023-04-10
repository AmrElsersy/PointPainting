import os
import cv2
import numpy as np
import argparse
import time
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from KittiCalibration import KittiCalibration
from visualizer import Visualizer
from BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from BiSeNetv2.utils.utils import preprocessing_kitti, postprocessing
from pointpainting import PointPainter
from bev_utils import boundary

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

class PaintLidarNode(Node):

    def __init__(self):
        super().__init__('paint_lidar_node')
        self.bridge = CvBridge()

        self.bisenetv2 = BiSeNetV2()
        self.checkpoint = torch.load('BiSeNetv2/checkpoints/BiseNetv2_150.pth', map_location=dev)
        self.bisenetv2.load_state_dict(self.checkpoint['bisenetv2'], strict=False)
        self.bisenetv2.eval()
        self.bisenetv2.to(device)

        self.painter = PointPainter()

        self.visualizer = Visualizer('2d')

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 1)
        self.create_subscription(PointCloud2, '/lidar/pointcloud', self.pointcloud_callback, 1)

        self.calib = KittiCalibration('Kitti_sample/calib/000038.txt')

        self.image_received = False
        self.image = None
        self.pointcloud_received = False
        self.pointcloud = None

    def image_callback(self, img_msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.image_received = True
        except CvBridgeError as e:
            print(e)

    def pointcloud_callback(self, pc_msg):
        self.pointcloud = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        self.pointcloud = np.array(list(self.pointcloud))
        self.pointcloud_received = True

    def process_data(self):
        if self.image_received and self.pointcloud_received:
            input_image = preprocessing_kitti(self.image)
            semantic = self.bisenetv2(input_image)
            semantic = postprocessing(semantic)

            painted_pointcloud = self.painter.paint(self.pointcloud, semantic, self.calib)

            if '2d' in self.visualizer.mode:
                color_image = self.visualizer.get_colored_image(self.image, semantic)
                scene_2D = self.visualizer.get_scene_2D(color_image, painted_pointcloud, self.calib)
                scene_2D = cv2.resize(scene_2D, (600, 900))
                cv2.imshow("scene", scene_2D)
                cv2.waitKey(1)

            self.image_received = False
            self.pointcloud_received = False


def main(args=None):
    rclpy.init(args=args)

    paint_lidar_node = PaintLidarNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(paint_lidar_node)
            paint