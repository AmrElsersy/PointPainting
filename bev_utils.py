import numpy as np

# =========================  Config ===============================
boundary = {
    "minX": 0,
    "maxX": 100,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

BEV_WIDTH  = 600
BEV_HEIGHT = 600


descretization_x = BEV_HEIGHT / boundary["maxX"]
descretization_y = BEV_WIDTH / (boundary["maxY"] - boundary["minY"])
descretization_z = 1 / float(np.abs(boundary['maxZ'] - boundary['minZ']))

# =========================== BEV RGB Map ==================================

def pointcloud_to_bev(pointcloud):
    assert pointcloud.shape[1] == 4
    
    pointcloud = clip_pointcloud(pointcloud)

    # sort by z ... to get the maximum z when using unique 
    # (as unique function gets the first unique elemnt so we attatch it with max value)
    z_indices = np.argsort(pointcloud[:,2])
    pointcloud = pointcloud[z_indices]

    MAP_HEIGHT = BEV_HEIGHT + 1
    MAP_WIDTH  = BEV_WIDTH  + 1

    height_map    = np.zeros((MAP_HEIGHT, MAP_WIDTH)) # max z
    intensity_map = np.zeros((MAP_HEIGHT, MAP_WIDTH)) # intensity (contains reflectivity or 1 if not supported)
    density_map   = np.zeros((MAP_HEIGHT, MAP_WIDTH)) # density of the mapped 3D points to a the pixel
    semantic_map  = np.ones((MAP_HEIGHT, MAP_WIDTH)) * 255 # semantic map default value 255 (which is background value) 

    # shape = (n_points, 1)
    x_bev = np.int_((BEV_HEIGHT)  - pointcloud[:, 0] * descretization_x )
    y_bev = np.int_((BEV_WIDTH/2) - pointcloud[:, 1] * descretization_y)
    z_bev = pointcloud[:, 2] 
    semantic_bev = pointcloud[:, 3]
    # print('semantic bev', np.unique(semantic_bev, return_counts=True))
    
    # shape = (n_points, 2)
    xy_bev = np.stack((x_bev, y_bev), axis=1)
    
    # xy_bev_unique.shape (n_unique_elements, 2)
    # indices.shape (n_unique_elements,) (needed for maximum Z)
    # counts.shape  (n_unique_elements,) .. counts is count of repeate times of each unique element (needed for density)
    xy_bev_unique, indices, counts = np.unique(xy_bev, axis=0, return_index=True, return_counts=True)

    # 1 or reflectivity if supported
    # intensity_map[x_bev_unique, y_bev_unique] = pointcloud[x_indices, 3]
    intensity_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = 1

    # points are sorted by z, so unique indices (first found indices) is the max z
    height_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = z_bev[indices]

    # density of points in each pixel
    density_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = np.minimum(1, np.log(counts + 1)/np.log(64) )

    semantic_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = semantic_bev[indices]

    # stack the BEV channels along 3rd axis
    BEV = np.dstack((intensity_map, height_map, density_map, semantic_map))
    return BEV

def clip_pointcloud(pointcloud):

    mask = np.where((pointcloud[:, 0] >= boundary["minX"]) & (pointcloud[:,0] <= boundary["maxX"]) & 
                    (pointcloud[:, 1] >= boundary["minY"]) & (pointcloud[:,1] <= boundary["maxY"])
                    # (pointcloud[:, 2] >= boundary["minZ"]) & (pointcloud[:,2] <= boundary["maxZ"])
    )

    return pointcloud[mask]