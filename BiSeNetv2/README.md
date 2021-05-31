# BiSeNetv2
#### Realtime Semantic Segmentation on KITTI Dataset
Trained on CityScapes Dataset & Finetuned on KITTI Semantic dataset


### Run Demo

```python
# weight_path is default to checkpoints/BiseNetv2_150.pth
python3 demo.py --path PATH_TO_IMAGE --weight_path PATH_TO_BISENET_WEIGHTS
```
![Screenshot 2021-06-01 00:36:28](https://user-images.githubusercontent.com/35613645/120246752-7847da80-c271-11eb-8c9a-1bf63a6c6ca5.png)


#### Visualize KITTI dataset
```python
python3 visualization.py
```

#### Finetuning
```python
python3 train.py --datapath PATH_TO_DATASET --pretrained PATH_TO_WEIGHTS_TO_CONTINUE --batch_size 16
```

#### Testing
```python
python3 test.py --weight_path PATH_TO_WEIGHTS

# you must have KITTI dataset in data/KITTI in the format below
```


### Structure
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


