# Depth Estimation using Self Supervised learning (Monocular images)


* [Environment Setup and Model Training](#env)
* [Experimentation Results](#results)
* [References](#ref)



<a name="env"></a>
# Environment Setup

### Using Conda 
```
conda env create -f depthestimate_env.yaml
conda activate depthestimate_env
```


Training your model
```
python main.py --conf configs/config.yaml 
```

To run in background on hpc

```
nohup python main.py --conf configs/config.yaml > output.log &
```

<a name="results"></a>
# Experimentation Results
| Encoder       | Architecture | Upsampling Mode | a1     | a2     | a3     | abs_rel | log_rms | rms   | sq_rel |
|---------------|--------------|-----------------|--------|--------|--------|---------|---------|-------|--------|
| ResNet50      | UNet         | bilinear        | 0.8744 | 0.9664 | 0.9878 | 0.123   | 0.1925  | 4.407 | 0.9378 |
| ResNet50      | UNet++       | bilinear        | 0.8801 | 0.9673 | 0.9896 | 0.1356  | 0.1852  | 4.348 | 0.9008 |
| ConvNext-tiny | UNet         | bilinear        | 0.9285 | 0.9701 | 0.9832 | 0.0996  | 0.1809  | 3.975 | 0.7534 |
| ConvNext-tiny | UNet         | ESPCN           | 0.8909 | 0.97   | 0.989  | 0.1017  | 0.185   | 3.886 | 0.587  |


Baseline Model (Monodepth2)|  ConvNext + UNet Implementation
:-------------------------:|:-------------------------:
![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/fig6.png)  |  ![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/WSP-2UP4_pred_convnext-unet_espcn-False.jpg)


<a name="ref"></a>
# References
<li>[1] Godard, Cl ́ement, et al., ”Digging into self-supervised monocular depth estimation.” Proceedings of the
IEEE/CVF International Conference on Computer Vision. 2019. arXiv:1806.01260
<li>[2] Source Code of Monodepth2: GitHub - nianticlabs/monodepth2: [ICCV 2019] Monocular depth estimation
from a single image.
<li>[3] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang, “UNet++: A Nested
U-Net Architecture for Medical Image Segmentation”. arXiv:1807.10165
<li>[4] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, “A Con-
vNet for the 2020s”. arXiv:2201.03545.
