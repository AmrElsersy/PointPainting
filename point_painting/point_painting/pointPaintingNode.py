import os
import numpy as np
import time
import torch
import rclpy
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, CompressedImage, PointField
from std_msgs.msg import String
from std_msgs.msg import Header

from point_painting.Calibration import Calibration
from point_painting.BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from point_painting.utils import preprocessing_kitti, postprocessing
from point_painting.pointpainting import PointPainter
from point_painting.visualizer import Visualizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

class PaintLidarNode(Node):

    def __init__(self):
        super().__init__('paint_lidar_node')

        # Segmantic Segmentation
        print('Loading BisenetV2 Model')
        self.bisenetv2 = BiSeNetV2()
        self.checkpoint = torch.load('/tmp/dev_ws/src/point_painting/point_painting/BiSeNetv2/checkpoints/BiseNetv2_150.pth', map_location=dev)
        self.bisenetv2.load_state_dict(self.checkpoint['bisenetv2'], strict=False)
        print('Loaded BisenetV2 Model')
        self.bisenetv2.eval()
        self.bisenetv2.to(device)
        print('Evaled BisenetV2 Model')

        # Fusion
        print('Starting Fusion')
        self.painter = PointPainter()
        print('Completed Fusion')

        # Visualizer
        self.visualizer = Visualizer()

        # Variables to store the incoming data
        self.image = None
        self.pointcloud = None
        self.calib = None

        # Subscribe
        self.image_subscription = self.create_subscription(CompressedImage, '/lucid_vision/camera_front/image_raw/compressed', self.image_callback, 10)
        self.lidar_subscription = self.create_subscription(PointCloud2, '/rslidar_points', self.lidar_callback, 10)
        self.camera_info_subscription = self.create_subscription(CameraInfo, '/lucid_vision/camera_front/camera_info', self.camera_info_callback, 10)
        print('Created Subscriber')

        # Publish
        self.painted_lidar_publisher = self.create_publisher(PointCloud2, '/painted_rslidar_points', 10)
        print('Created Publisher')

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.process_data()

    def lidar_callback(self, msg):
        self.pointcloud = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        self.process_data()

    def camera_info_callback(self, msg):
        self.P2 = np.array(msg.p).reshape(3, 4)
        self.R0_rect = np.array(msg.r).reshape(3, 3)
        calib_path = '/tmp/dev_ws/src/point_painting/point_painting/top_center_lidar-to-center_camera-extrinsic.json'
        self.calib = Calibration(calib_path, self.P2, self.R0_rect, from_json=True)
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
        painted_lidar_msg.header.frame_id = 'rslidar'
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

        # Publish Painted Lidar
        self.painted_lidar_publisher.publish(painted_lidar_msg)
        t5 = time.time()

        # Add Visualizer
        color_image = self.visualizer.get_colored_image(self.image, semantic)
        scene_2D = self.visualizer.get_scene_2D(color_image, painted_pointcloud, self.calib)
        scene_2D = cv2.resize(scene_2D, (600, 900))
        cv2.imshow("scene", scene_2D)
        cv2.waitKey(1)
        t6 = time.time()

        print(f'Time of bisenetv2 = {1000 * (t2-t1)} ms')
        print(f'Time of postprocesssing = {1000 * (t3-t2)} ms')
        print(f'Time of pointpainting = {1000 * (t4-t3)} ms')
        print(f'Time of publishing = {1000 * (t5-t4)} ms')
        print(f'Time of pointpainting = {1000 * (t6-t5)} ms')
        print(f'Time of Total = {1000 * (t6-t1)} ms')

def main(args=None):
    rclpy.init(args=args)

    # if calib file is in kitti video format
    # calib = Calibration(args.calib_path, from_video=True)
    # if calib file is in normal kitti format
    # TO DO !!: ADD PATH here

    paint_lidar_node = PaintLidarNode()

    rclpy.spin(paint_lidar_node)
    
    PaintLidarNode.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    main()

