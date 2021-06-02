[![Made withPython](https://img.shields.io/badge/Made%20with-python-407eaf?style=for-the-badge&logo=python)](https://www.python.org/)
[![Made withPytorch](https://img.shields.io/badge/Made%20with-pytorch-ee4c2c?style=for-the-badge&logo=pytorch)](https://www.pytorch.org/)
# PointPainting-Semantic-Segmentation

My Pytorch implementation of [PointPainting Paper](https://arxiv.org/abs/1911.10150) for realtime pointcloud semantic segmentation painting (labeling each point with a class) based on semantic segmentation maps using [BiSeNetv2](https://arxiv.org/abs/1808.00897) 

![pointpainting](images/point_painting.png)


#### Project

- BiSeNetv2 model trained on KITTI dataset 
- Implementation of the pointpainting fusion algorithm

### Demo Video
![Demo](images/video.gif)

### Download the checkpoint
[Download from Drive](https://drive.google.com/file/d/10-WxqSmyFKW72_1D-2vwu7BzUFlCOwgb/view?usp=sharing)
Place it in "BiSeNetv2/checkpoints"

### Run Demo
```python
python3 demo.py --image_path PATH_TO_IMAGE --pointcloud_path PATH_TO_POINTCLOUD --calib_path PATH_TO_CALIB --weights_path PATH_TO_MODEL_WEIGHTS

# note that default arguments are set to one of uploaded kitti samples so you can run it as
python3 demo.py
```
![output_2d_demo](images/2d.png)

```python
# add --mode 3d to see a 3d visualization of painted pointcloud
python3 demo.py --mode 3d
```
![output_3d_demo](images/3d.gif)


### Run Demo on Kitti Videos
Kitti Provides sequential videos for testing, Download them from [Kitti Videos]() by downloading video data(left & pointclouds)(download from [synced+rectified data]) and calibs (download from [calibration])
in the selected video


```python
# PATH_TO_VIDEO is path contains 'image_02' & 'velodyne_points' together
# PATH_TO_CALIB is path contains calib files ['calib_cam_to_cam', '', '']
# mode 2d to visualize image+bev .. 3d to visualize 3d painted pointcloud
python3 demo_video.py --video_path PATH_TO_VIDEO --calib_path PATH_TO_CALIB --mode 3d
```
![video_3d](images/video_3d.gif)


## BiSeNetv2
Realtime semantic segmentation on images

![model](images/model.png)

Thanks to https://github.com/CoinCheung/BiSeNet for the implementation trained on CityScapes datasets.
I used it and finetuned it on KITTI dataset using Pytorch

### Training on KITTI dataset
```python
cd BiSeNetv2
python3 train.py 
```
I trained it on Colab and provided the [notebook](https://github.com/AmrElsersy/PointPainting/blob/master/BiSeNetv2/BiseNet_Train.ipynb) 

![training](images/training.png)

### Test on KITTI Semantic
```python
cd BiSeNetv2
python3 test.py 
```

### KITTI Dataset
Semantic KITTI dataset contains 200 images for training & 200 for testing <br>
Download it from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)

```python
# visualize dataset on tensorboard
python3 visualization.py --tensorboard

# PATH_TO_TENSORBOARD_FOLDER is path "BiSeNetv2/checkpoints/tensorboard/"
tensorboard --logdir PATH_TO_TENSORBOARD_FOLDER
```
![tensorboard](images/tensorboard.png)


#### Folder structure    
    ├── BiSeNetv2
	    ├── checkpoints
    		├── BiseNetv2_150.pth 	# path to model
		    ├── tensorboard 		# path to save tensorboard events
	    ├── data 					# path to kitti semantic dataset
	        ├── KITTI
              ├── testing
	              ├── image_2
              ├── training
                ├── image_2
                ├── instance
                ├── semantic
                ├── semantic_rgb
	    
        ├── utils
          ├── label.py 				# label information (colors/ids/names)
          ├── utils.py 				# utils functions
        ├── train.py
        ├── test.py
        
    ├── Kitti_sample				# 2 images & pointclouds & calib for testing (by demo.py)
    ├── KittiCalibration.py 		# Stores Calibration file matrices 
    ├── KittiVideo.py 				# Kitti Video Reader
    ├── bev_utils.py 				# BEV algorithms
    ├── demo.py 					# demo to test on 1 sample (Kitti_sample)
    ├── demo_video.py 				# demo to test on Kitti Videos
    ├── pointpainting.py 			# implementation of PointPainting
    ├── visualizer.py 				# visualizer using opend3d & opencv


### References
- My Review https://docs.google.com/document/d/1AtpbLfCl_uL5BpwlYDgpdM_2WtCIucDrRomFjSp6bhg/edit?usp=sharing

- Semantic Seg Overview :https://medium.com/beyondminds/a-simple-guide-to-semantic-segmentation-effcf83e7e54 .. https://medium.com/swlh/understanding-multi-scale-representation-learning-architectures-in-convolutional-neural-networks-a71497d1e07c

- Conv Types (Atrous, Transposed) https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

- DeepLabv3 https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74

- Survay: Image Segmentation using Deep Learning https://arxiv.org/pdf/2001.05566.pdf
- Survay Article https://medium.com/swlh/image-segmentation-using-deep-learning-a-survey-e37e0f0a1489

- BiseNet (understanding spatial & context information) (good) https://prince-canuma.medium.com/spatial-path-context-path-3f03ed0c0cf5 .. https://medium.datadriveninvestor.com/bisenet-for-real-time-segmentation-part-i-bf8c04afc448

- Receptive field https://blog.christianperone.com/2017/11/the-effective-receptive-field-on-cnns/	
