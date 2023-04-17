# Building the Image

```
sudo docker build -t pointpaintingvideo_ros .
```

# Building the Container

```
sudo rocker --nvidia --x11 --env NVIDIA_DRIVER_CAPABILITIES=all --volume ~/Documents/thesis/PointPaintingVideo:/tmp/PointPainting -- pointpaintingvideo
```

```
sudo rocker --nvidia --x11 --env NVIDIA_DRIVER_CAPABILITIES=all --volume /mnt/f/git/PointPaintingVideo/point_painting:/tmp/dev_ws/src/point_painting /mnt/f/git/PointPaintingVideo/rosbag:/tmp/dev_ws/src/point_painting/rosbag -- pointpaintingvideo_ros
```


# Source Ros
```
source "/opt/ros/humble/setup.bash"
```

# Update
```
sudo apt update
```

# Install compressed images
```
sudo apt install ros-humble-image-transport-plugins
```

# Create project
```
ros2 pkg create --build-type ament_python pointPaintingRos
```

# Build all projects
```
colcon build
```

# Build selected project
```
colcon build --packages-select
```

# source Projects
```
source install/setup.bash
```

# Play rosbag
```
ros2 bag play rosbag2_2023_03_16-09_05_41_0.db3
```

# Launch projet
```
ros2 launch point_painting point_painting_launch.py
```

# Open New Terminal in docker
```
docker exec -it
```

# Run image viewer
```
ros2 run rqt_image_view rqt_image_view
```

# Run topic viewer
```
ros2 topic echo /lucid_vision/camera_front/camera_info
```