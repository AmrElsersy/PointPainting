import os
import numpy as np
import time
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from std_msgs.msg import Header
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

        # Segmantic Segmentation
        self.bisenetv2 = BiSeNetV2()
        self.checkpoint = torch.load('BiSeNetv2/checkpoints/BiseNetv2_150.pth', map_location=dev)
        self.bisenetv2.load_state_dict(self.checkpoint['bisenetv2'], strict=False)
        self.bisenetv2.eval()
        self.bisenetv2.to(device)

        # Fusion
        self.painter = PointPainter()

        # Variables to store the incoming data
        self.image = None
        self.pointcloud = None
        self.calib = None

        # Subscribe
        self.image_subscription = self.create_subscription(Image, 'image_topic', self.image_callback, 10)
        self.lidar_subscription = self.create_subscription(PointCloud2, 'lidar_topic', self.lidar_callback, 10)
        self.config_subscription = self.create_subscription(String, 'config_topic', self.config_callback, 10)

        # Publish
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

    paint_lidar_node = PaintLidarNode()

    rclpy.spin(paint_lidar_node)
    
    PaintLidarNode.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    main()

