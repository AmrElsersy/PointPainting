import os
import cv2
import numpy as np

class KittiVideo:
    """ Load data for KITTI videos """

    def __init__(self, imgL_dir, imgR_dir, lidar_dir, calib_dir):
        self.calib = KittiCalibration(calib_path=calib_dir, from_video=True)
        self.imgL_dir = imgL_dir
        self.imgR_dir = imgR_dir
        self.lidar_dir = lidar_dir

        self.imgL_filenames = sorted(
            [os.path.join(imgL_dir, filename) for filename in os.listdir(imgL_dir)]
        )
        
        self.imgR_filenames = sorted(
            [os.path.join(imgR_dir, filename) for filename in os.listdir(imgR_dir)]
        )

        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )

        assert(len(self.imgL_filenames) == len(self.imgR_filenames))
        assert(len(self.imgL_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.imgL_filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        imgL, imgR = self.__get_image(index) 
        return imgL, imgR, self.__get_lidar(index), self.__get_calibration()

    def __get_image(self, index):
        assert index < self.num_samples
        imgL_filename = self.imgL_filenames[index]
        imgR_filename = self.imgR_filenames[index]
        return cv2.imread(imgL_filename), cv2.imread(imgR_filename)
    
    def __get_lidar(self, index):
        assert index < self.num_samples
        lidar_filename = self.lidar_filenames[index]

        scan = np.fromfile(lidar_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def __get_calibration(self):
        return self.calib