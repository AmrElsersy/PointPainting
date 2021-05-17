import os
import cv2
import numpy as np
import KittiCalibration

class KittiVideo:
    """ Load data for KITTI videos """

    def __init__(self, video_root, calib_root):
        self.video_root = video_root
        self.calib_root = calib_root

        self.calib = KittiCalibration(calib_path=calib_root, from_video=True)
        self.images_dir = os.path.join(self.video_root, 'image_2')
        self.lidar_dir = os.path.join(self.video_root, 'velodyne')

        self.images_filenames = sorted(
            [os.path.join(self.images_dir, filename) for filename in os.listdir(self.images_dir)]
        )

        self.lidar_filenames = sorted(
            [os.path.join(self.lidar_dir, filename) for filename in os.listdir(self.lidar_dir)]
        )

        assert(len(self.images_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.imgL_filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = self.__get_image(index) 
        return image, self.__get_lidar(index), self.__get_calibration()

    def __get_image(self, index):
        assert index < self.num_samples
        path = self.images_dir[index]
        return cv2.imread(path)
    
    def __get_lidar(self, index):
        assert index < self.num_samples
        path = self.lidar_filenames[index]
        return np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    def __get_calibration(self):
        return self.calib