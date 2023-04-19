import os
import cv2
import numpy as np
from Calibration import Calibration

class Video:
    """ Load data for KITTI videos """

    def __init__(self, video_root, calib_root):
        self.video_root = video_root
        self.calib_root = calib_root

        self.calib = Calibration(calib_path=calib_root, from_video=True)
        self.images_dir = os.path.join(self.video_root, 'image_02/data')
        self.lidar_dir = os.path.join(self.video_root, 'velodyne_points/data')

        self.images_filenames = sorted(
            [os.path.join(self.images_dir, filename) for filename in os.listdir(self.images_dir)]
        )

        self.lidar_filenames = sorted(
            [os.path.join(self.lidar_dir, filename) for filename in os.listdir(self.lidar_dir)]
        )

        self.num_samples = min(len(self.images_filenames), len(self.lidar_filenames))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = self.__get_image(index) 
        return image, self.__get_lidar(index), self.__get_calibration()

    def __get_image(self, index):
        assert index < self.num_samples
        path = self.images_filenames[index]
        return cv2.imread(path)
    
    def __get_lidar(self, index):
        assert index < self.num_samples
        path = self.lidar_filenames[index]
        return np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    def __get_calibration(self):
        return self.calib