# ACM MM2024 3rd Workshop on QoEVMA: ViSam-PCQA
Official repo for '[Visual-Saliency Guided Multi-modal Learning for No Reference Point Cloud Quality Assessment](https://dl.acm.org/doi/10.1145/3689093.3689183)'.
## Abstract
As 3D immersive media continues to gain prominence, Point Cloud Quality Assessment (PCQA) is essential for ensuring high-quality user experiences. This paper introduces ViSam-PCQA, a no-reference PCQA metric guided by saliency information across three modalities, which facilitates the performance of the quality prediction. Firstly, we project the 3D point cloud to acquire 2D texture, depth, and normal maps. Secondly, we extract the saliency map based on the texture map and refine it with the corresponding depth map. This refined saliency map is used to weight low-level feature maps to highlight perceptually important areas in the texture channel. Thirdly, high-level features from the texture, normal, and depth maps are then processed by a Transformer to capture global and local point cloud representations across modalities. Lastly, saliency along with global and local embeddings, are concatenated and processed through a multi-task decoder to derive the final quality scores. Our experiments on the SJTU, WPC, and BASICS datasets show high Spearman rank order correlation coefficients/Pearson linear correlation coefficients of 0.953/0.962, 0.920/0.920 and 0.887/0.936 respectively, demonstrating superior performance compared to current state-of-the-art methods.

## Motivation

<img src="https://github.com/cwi-dis/ViSam-PCQA_MM2024Workshop/blob/master/imgs/motivation.jpg" align="left" />

**Illustration to show the perceptual impact of distortion in different areas on redandblack point cloud.** (a) is the reference version.   
(b)-(d) depict the effects of introducing geometry and color Gaussian noise with equal intensity on the face, dress, and legs, respectively.  
Notably, (c) exhibits nearly identical perceptual quality as the reference point clouds, attributed to the chaotic background texture that effectively
masks the distortion. (d) ranks second in perceptual quality, while (a) is observed to have the least favorable perceptual quality.  
The perceptual quality of point clouds is dependent on distortion type since the HVS has different tolerances for different distortions, and where the distortion is located can have a huge impact on the overall quality of point clouds.

## Framework

<p align="center">
  <img src="https://github.com/cwi-dis/ViSam-PCQA_MM2024Workshop/blob/master/imgs/framework.jpg" /> 
</p>

The framework overview is exhibited as above. The point clouds are first projected into three different modalities, texture map, depth map and normal map. Then we use an image encoder 𝜃𝐼 to extract the low-level features and high-level features, respectively. Since the primary cues for visual attention often come from the 2D projections captured by the retina, depth and normal information are crucial for spatial perception and object localization. We use a pre-trained visual saliency model on the texture image and use its output to correct the low-level feature after an image encoder. At the same time, the texture image, depth image and normal image are put into the same image encoder to get the semantic feature. Subsequently, the semantic features are put into an intra-and-inter modality attention module to get the global and local features of the point cloud. Finally, the global and local features are concatenated to the distortion type classification branch to learn the distortion-oriented features. The corrected visual saliency with the distortion-oriented features are concatenated and decoded into the quality values via the quality regression branch.

# How to run the code 
## Environment Build

We train and test the code on the Ubuntu 18.04 platform with open3d and python=3.7. 
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
The GPU is A100 with 48 GB memory,  batchsize = 18.

## Begin training

You can simply train the M3-Unity by referring to train.sh. For example, train M3-Unity on the BASICS dataset with the following command:

```
python -u train_img.py \
--learning_rate 0.00005 \
--model MM_PCQA \
--batch_size 18 \
--database BASICS  \
--data_dir_texture_img path to basic_projections/ \
--data_dir_depth_img path to basic_depth_maps/ \
--data_dir_normal_img path to basic_normal_maps/ \
--data_dir_vs_img path to basic_visual saliency maps/ \
--loss l2rank \
--num_epochs 100 \
--k_fold_num 1 \
--use_classificaiton 1 \
--use_local 1 \
--method_label Baseline
```

 **The training data of the projections and patches, will be accessed soon.**  

# Visual Saliency Impact Visualization
<p align="left">
  <img src="https://github.com/cwi-dis/ViSam-PCQA_MM2024Workshop/blob/master/imgs/visualsaliency.jpg" /> 
</p>

# Performance
<p align="left">
  <img src="https://github.com/cwi-dis/ViSam-PCQA_MM2024Workshop/blob/master/imgs/Performance.png" /> 
</p>

# Bibtex 
-----------
If you find our code useful code please cite the paper   
```
@inproceedings{10.1145/3689093.3689183,
author = {Zhou, Xuemei and Viola, Irene and Yin, Ruihong and Cesar, Pablo},
title = {Visual-Saliency Guided Multi-modal Learning for No Reference Point Cloud Quality Assessment},
year = {2024},
isbn = {9798400712043},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3689093.3689183},
doi = {10.1145/3689093.3689183},
booktitle = {Proceedings of the 3rd Workshop on Quality of Experience in Visual Multimedia Applications},
pages = {39–47},
numpages = {9},
keywords = {multi-modal, no reference, point cloud quality assessment, projection, visual saliency},
location = {Melbourne VIC, Australia},
series = {QoEVMA'24}
}
```
If you encounter any issues with the code or training dataset, please contact xuemei.zhou@cwi.nl
