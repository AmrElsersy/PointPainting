# Building the Image

```
sudo docker build -t pointpaintingvideo .
```

# Building the Container

```
sudo rocker --nvidia --x11 --env NVIDIA_DRIVER_CAPABILITIES=all --volume ~/Documents/thesis/PointPaintingVideo:/tmp/PointPainting -- pointpaintingvideo
```

python3 demo_video.py --video_path KITTI/1 --calib_path KITTI/1/calib --mode 2d
python3 demo_video.py --video_path KITTI/2 --calib_path KITTI/2/calib --mode 2d
python3 demo_video.py --video_path KITTI/3 --calib_path KITTI/3/calib --mode 2d

python3 demo.py --image_path Kitti_Video/1/image_02/data --pointcloud_path Kitti_Video/1/velodyne_points/data --calib_path Kitti_Video/1/calib --weights_path BiSeNetv2/checkpoints/BiseNetv2_150.pth

