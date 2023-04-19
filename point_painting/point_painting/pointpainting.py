"""

"""

import cv2
import time
import numpy as np
from point_painting import Calibration
from point_painting.bev_utils import clip_pointcloud

class PointPainter():
    def __init__(self):
        self.image_shape = (1242, 375)

    def paint(self, pointcloud, semantic, calib:Calibration):
        """
            Point Painting Fusion Algorithm
            Specify a class for each point in the pointcloud
            Arguments:
                pointcloud: shape = [n_points, 3]
                semantic: semantic map of left camera .. shape [1247, 375]
                calib: Calibration object to access camera matrices
            Return:
                semantic/painted pointcloud of shape [n_points, 4] .. additional channel indicates class
        """
        pointcloud = clip_pointcloud(pointcloud)
        t1 = time.time()
        semantic = cv2.resize(semantic, self.image_shape, interpolation=cv2.INTER_NEAREST)

        ### Project all the pointcloud onto the image

        # Num of points
        N = pointcloud.shape[0]
        # remove reflectivity
        pointcloud = pointcloud[:,:3] 
        # add homoginous 1 .. pointcloud shape (N, 4)
        pointcloud = np.hstack((pointcloud, np.ones((N, 1), dtype=np.float32)))

        # Extrinsic from (3,4) to (4,4)
        velo_to_cam = np.vstack((calib.Tr_velo_to_cam, np.array([0,0,0,1])))
        # Rotation R from (3,3) to (4,4)
        R_rect = np.zeros((4,4))
        R_rect[:3,:3] = calib.R0_rect
        R_rect[3,3] = 1
        # Projection matrix from (3,4) to (4,4)
        P2 = np.vstack((calib.P2, np.array([0,0,0,1])))
        # print(velo_to_cam.shape, R_rect.shape, P2.shape)

        # 2D image point = Projection @ Rectification @ Extrinsic @ 3D Velodyne
        projected_points = P2 @ R_rect @ velo_to_cam @ pointcloud.T # (4,N) == (4,4) @ (4,N) 
        # device by z homoginous to get x & y
        projected_points /= projected_points[2]
        # get only x & y (screen coord)
        projected_points = projected_points[:2].T # (N, 2)
        projected_points = projected_points.astype(np.int32)

        ### Filter 
        # filter is the index that contains background value (255) to be colorless
        # semantic map dosn't garentee that it has value of 255 so we set [0,0] index to 255
        semantic[0,0] = 255
        filter = 0

        # mask to remove any point that has x or y > image coord
        x_mask = projected_points[:,0] > self.image_shape[0] - 1  
        y_mask = projected_points[:,1] > self.image_shape[1] - 1 
        # mask to remove neg values
        neg_mask = projected_points < 0
        neg_mask = np.any(neg_mask, axis=1)

        # filter points that are outside image coord (assign it to filter value (which leads to background semantic class))        
        projected_points[x_mask] = filter
        projected_points[y_mask] = filter
        projected_points[neg_mask] = filter

        # semantic channel that has class for each point 
        semantic_channel = np.zeros((N,1), dtype=np.float32)
        # could be accessed by array of y values & array of x values (similar to looping with [x,y])
        semantic_channel = semantic[projected_points[:,1], projected_points[:,0]].reshape(-1,1)

        # The line of assigning semantic_channel is equivilent to looping to set each point but looping is slow
        # for i in range(N):
        #     x = projected_points[i,0]
        #     y = projected_points[i,1]
        #     semantic_channel[i] = semantic[y,x]

        t = time.time() - t1
        # print('time', t)
        # print(np.unique(semantic_channel, return_counts=True))

        painted_pointcloud = np.hstack((pointcloud[:,:3], semantic_channel)) # (N, 4)
        return painted_pointcloud


if __name__ == '__main__':
    x = np.array([
        [0., 1., 2.],
        [3., 4., 5.],
        [6., 7., 8.]])

    mask = np.where(x>5)

    print(mask)
    print(x[mask])