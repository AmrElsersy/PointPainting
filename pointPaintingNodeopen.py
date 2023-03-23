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

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
import rclpy
from rclpy.node import Node

class PointPaintingNode(Node):
    def __init__(self, args):
        super().__init__('point_painting_node')

        self.args = args

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.dev)

        self.bisenetv2 = BiSeNetV2()
        checkpoint = torch.load(args.weights_path, map_location=self.dev)
        self.bisenetv2.load_state_dict(checkpoint['bisenetv2'], strict=False)
        self.bisenetv2.eval()
        self.bisenetv2.to(self.device)

        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.TRT_LOGGER)

        self.painter = PointPainter()
        self.visualizer = Visualizer(args.mode)

        self.image = None
        self.pointcloud = None
        self.calib = None

        self.image_subscription = self.create_subscription(Image, 'image_topic', self.image_callback, 10)
        self.lidar_subscription = self.create_subscription(PointCloud2, 'lidar_topic', self.lidar_callback, 10)
        self.config_subscription = self.create_subscription(String, 'config_topic', self.config_callback, 10)

        self.painted_lidar_publisher = self.create_publisher(PointCloud2, 'painted_lidar_topic', 10)

    def image_callback(self, msg):
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.process_data()

    def lidar_callback(self, msg):
        self.pointcloud = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        self.process_data()

    def config_callback(self, msg):
        self.calib = KittiCalibration(msg.data)
        self.process_data()

    def process_data(self):
        if self.image is None or self.pointcloud is None or self.calib is None:
            return

        t1 = time.time()
        input_image = preprocessing_kitti(self.image)
        semantic = self.bisenetv2(input_image)
        t2 = time.time()
        semantic = postprocessing(semantic)
        t3 = time.time()

        painted_pointcloud = self.painter.paint(self.pointcloud, semantic, self.calib)
        t4 = time.time()

        # Publish the painted_pointcloud as a PointCloud2 message
        painted_lidar_msg = PointCloud2()
        # Set the header and data fields of the painted_lidar_msg
        painted_lidar_msg.header.stamp = self.get_clock().now().to_msg()
        painted_lidar_msg.header.frame_id = 'painted_lidar'
        painted_lidar_msg.height = 1
        painted_lidar_msg.width = painted_pointcloud.shape[0]
        painted_lidar_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        painted_lidar_msg.is_bigendian = False
        painted_lidar_msg.point_step = 16
        painted_lidar_msg.row_step = painted_lidar_msg.point_step * painted_lidar_msg.width
        painted_lidar_msg.is_dense = True
        painted_lidar_msg.data = painted_pointcloud.astype(np.float32).tobytes()

        self.painted_lidar_publisher.publish(painted_lidar_msg)

        print(f'Time of bisenetv2 = {1000 * (t2-t1)} ms')
        print(f'Time of postprocesssing = {1000 * (t3-t2)} ms')
        print(f'Time of pointpainting = {1000 * (t4-t3)} ms')
        print(f'Time of Total = {1000 * (t4-t1)} ms')

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='BiSeNetv2/checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'],
                        help='visualization mode .. img is semantic image .. 2d is semantic + bev .. 3d is colored pointcloud')

    args = parser.parse_args(args=args)

    point_painting_node = PointPaintingNode(args)

    try:
        rclpy.spin(point_painting_node)
    except KeyboardInterrupt:
        pass
    finally:
        point_painting_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

