import os
import cv2
import numpy as np

class KittiCalibration:
    """
        Perform different types of calibration between camera & LIDAR
        image = Projection * Camera3D_after_rectification
        image = Projection * R_Rectification * Camera3D_reference
    """
    def __init__(self, calib_path, from_video=False):
        self.calib_path = calib_path
        self.calib_matrix = {}
        if from_video:
            self.calib_matrix = self.parse_calib_from_video(calib_path)
            self.calib_path = os.path.join(calib_path, "modified_calib_file.txt")
            print('#################', self.calib_path)
        else:
            self.calib_matrix = self.parse_calib_files(calib_path)

        self.P0 = self.calib_matrix["P0"]
        self.P1 = self.calib_matrix["P1"]
        # Projection Matrix (Intrensic) .. from camera 3d (after rectification) to image coord.
        self.P2 = self.calib_matrix["P2"].reshape(3, 4)
        self.P3 = self.calib_matrix["P3"]
        # rectification rotation matrix 3x3
        self.R0_rect = self.calib_matrix["R0_rect"].reshape(3,3)
        # Extrensic Transilation-Rotation Matrix from LIDAR to Cam ref(before rectification)
        self.Tr_velo_to_cam = self.calib_matrix["Tr_velo_to_cam"].reshape(3,4)
        # inverse of Tr_velo_cam
        self.Tr_cam_to_velo = self.inverse_Tr(self.Tr_velo_to_cam)

    def parse_calib_files(self, calib_path):
        assert self.calib_path is not None

        mat_ = {}
        with open(os.path.join(calib_path), 'r') as calib_file:
            for line in calib_file:
                line = line.split()
                # Avoiding empty line exception
                if len(line) == 0:
                    continue
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try: 
                    mat_[line[0][:-1]] = np.array(line[1:], dtype=np.float32)
                except ValueError:
                    continue

        return mat_

    def parse_calib_from_video(self, calib_path):
        """ Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        """
        assert calib_path is not None

        mat_ = {}
        cam2cam = self.parse_calib_files(
            os.path.join(calib_path, "calib_cam_to_cam.txt")
        )
        velo2cam = self.parse_calib_files(
            os.path.join(calib_path, "calib_velo_to_cam.txt")
        )


        mat_["P0"] = cam2cam["P_rect_00"]
        mat_["P1"] = cam2cam["P_rect_01"]
        mat_["P2"] = cam2cam["P_rect_02"]
        mat_["P3"] = cam2cam["P_rect_03"]

        mat_["R0_rect"] = cam2cam["R_rect_00"]

        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        mat_["Tr_velo_to_cam"] = Tr_velo_to_cam
        return mat_


    def rotx(self, t):
        """ 3D Rotation about the x-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def roty(self, t):
        """ Rotation about the y-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rotz(self, t):
        """ Rotation about the z-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([
            [c,-s, 0], 
            [s, c, 0], 
            [0, 0, 1]])

    def project_lidar_to_image(self, corners=None):
        """
            Projecting a tensor of objects to image plane (u,v)
            P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                        0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                        0,      0,      1,      0]
            image2 coord:
             ----> x-axis (u)
            |
            |
            v y-axis (v)
            velodyne coord:
            front x, left y, up z
            rect/ref camera coord:
            right x, down y, front z
            Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        """
        assert corners is not None

        P_2 = self.P2
        # We're using R_0 as a reference rotation
        # R_0 is expanded to (4,4) by adding a new row & col with zeros & R_0(4,4) = 1
        R_0 = self.R0_rect
        R_0 = np.pad(R_0, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        R_0[-1, -1] = 1

        # Extrinsic transformation matrix from velodyne to camera coordinates
        T_velo_cam = self.Tr_velo_to_cam
        T_velo_cam = np.pad(T_velo_cam, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        T_velo_cam[-1, -1] = 1

        projected_corners = []
        for corner in corners:
            # y_homogeneous = P_2.R_0.T_velo_cam.X_homogeneous
            corner = np.array([corner[0], corner[1], corner[2], 1])
            y = np.dot(T_velo_cam, corner)
            y = np.dot(R_0, y)
            y = np.dot(P_2, y)
            # Updating x, y & z from homogeneous to non homogeneous (u,v)
            z = y[2]
            x = y[0] / z
            y = y[1] / z
            projected_corners.append([x, y, z])

        return np.array(projected_corners, dtype=np.float32)

    def inverse_Tr(self, T):
        """ 
            get inverse of Translation Rotation 4x4 Matrix
            Args:
                T: 4x4 Matrix
                    ([
                        [R(3x3) t],
                        [0 0 0  1]
                    ])
            Return:
                Inverse: 4x4 Matrix
                    ([
                        [R^-1   -R^1 * t],
                        [0 0 0         1]
                    ])                
        """
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        # print(R.shape, t.shape)
        R_inv = np.linalg.inv(R)
        t_inv = np.dot(-R_inv, t).reshape(3,1)
        T_inv = np.hstack((R_inv, t_inv))
        T_inv = np.vstack( (T_inv, np.array([0,0,0,1])) )
        return T_inv

    def rectified_camera_to_velodyne(self, points):
        """
            Converts 3D Box in Camera coordinates(after rectification) to 3D Velodyne coordinates
            Args: points
                numpy array (N, 3) in cam coord, N is points number 
            return: 
                numpy array (N, 3) in velo coord.
        """

        # from rectified cam to ref cam 3d
        R_rect_inv = np.linalg.inv(self.R0_rect)
        points_ref =  np.dot(R_rect_inv, points.T) # 3xN

        # add homogenious 4xN
        points_ref = np.vstack((points_ref, np.ones((1, points_ref.shape[1]), dtype=np.float)))

        # velodyne = ref_to_velo * points_ref
        points_3d_velodyne = np.dot(self.Tr_cam_to_velo, points_ref)

        return points_3d_velodyne.T
