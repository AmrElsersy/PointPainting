# Building the Image

```
sudo docker build -t pointpaintingvideo_ros .
```

# Building the Container

```
sudo rocker --nvidia --x11 --env NVIDIA_DRIVER_CAPABILITIES=all --volume ~/Documents/thesis/PointPaintingVideo:/tmp/PointPainting -- pointpaintingvideo
```

```
sudo rocker --nvidia --x11 --env NVIDIA_DRIVER_CAPABILITIES=all --volume /mnt/f/git/PointPaintingVideo:/tmp/PointPainting -- pointpaintingvideo_ros
```

python3 demo_video.py --video_path KITTI/1 --calib_path KITTI/1/calib --mode 2d
python3 demo_video.py --video_path KITTI/2 --calib_path KITTI/2/calib --mode 2d
python3 demo_video.py --video_path KITTI/3 --calib_path KITTI/3/calib --mode 2d

python3 demo.py --image_path Kitti_Video/1/image_02/data --pointcloud_path Kitti_Video/1/velodyne_points/data --calib_path Kitti_Video/1/calib --weights_path BiSeNetv2/checkpoints/BiseNetv2_150.pth


source ros
source "/opt/ros/humble/setup.bash"





Play rosbag
ros2 bag play rosbag2_2023_03_16-09_05_41_0.db3

update
sudo apt update

install compressed images
sudo apt install ros-humble-image-transport-plugins



source projects
source install/setup.bash

build all
colcon build

build project
colcon build --packages-select

create project
ros2 pkg create --build-type ament_python pointPaintingRos

launch projet
ros2 launch point_painting point_painting_launch.py

run image viewer
ros2 run rqt_image_view rqt_image_view

run topic viewer
ros2 topic echo /lucid_vision/camera_front/camera_info