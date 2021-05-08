# PointPainting-Semantic-Segmentation



### References
- My Review
https://docs.google.com/document/d/1AtpbLfCl_uL5BpwlYDgpdM_2WtCIucDrRomFjSp6bhg/edit

- Semantic Seg Overview Ray2 
https://medium.com/beyondminds/a-simple-guide-to-semantic-segmentation-effcf83e7e54
https://medium.com/swlh/understanding-multi-scale-representation-learning-architectures-in-convolutional-neural-networks-a71497d1e07c

- Conv Types (Atrous, Transposed)
https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

- DeepLabv3
https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74

- Losses
	Cross Entropy (bad due to high loss to background)
	Focal Loss (makes background loss = 0)
	Dice Loss (calculate overlap between predicted semantic map & gt)

- Data Augumentation
	Some typical transformations include translation, reflection, rotation, warping, scaling,
	color space shifting, cropping, and projections onto principal components.

- Survay: Image Segmentation using Deep Learning
https://arxiv.org/pdf/2001.05566.pdf
- Survay Article
https://medium.com/swlh/image-segmentation-using-deep-learning-a-survey-e37e0f0a1489

- BiseNet (understanding spatial & context information) (good)
https://prince-canuma.medium.com/spatial-path-context-path-3f03ed0c0cf5
https://medium.datadriveninvestor.com/bisenet-for-real-time-segmentation-part-i-bf8c04afc448

- Receptive field
https://blog.christianperone.com/2017/11/the-effective-receptive-field-on-cnns/	
