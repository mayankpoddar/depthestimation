<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://rehost.in/templates">
    <img src="https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo.gif" alt="testVideo" width="450" height="300">
    <img src="https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-baseline-resnet-unet.gif" alt="baseline" width="450" height="300">
  </a>

<h3 align="center">Depth Estimation using Self Supervised learning</h3>
  <p align="center">
    an extension of "Digging into self-supervised monocular depth estimation"
    <br />
    <a href="https://www.tensorflow.org/tensorboard"><strong>using Tensorboard »</strong></a>
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

Keeping Monodepth2[1] as our baseline model, we propose certain architectural changes that
improve the performance of Monodepth V2 by incorporating recent developments for convolutional
neural networks and using a common encoder backbone. In the next phase, we plan to incorporate
NYUv2 dataset and experiment with various augmentation techniques to further improve the
performance on the optimal backbone and architecture selected. All the experiments are performed
on the KITTI dataset [5] and the NYUv2 dataset [6].

* [Environment Setup and Model Training](#env)
* [Experimentation Results](#results)
* [References](#ref)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

<a name="env"></a>

## Environment Setup

1. Install Conda:

```
conda env create -f depthestimate_env.yaml
conda activate depthestimate_env
```

### Usage

1. Train Model

Training your model
```
python main.py --conf configs/config.yaml 
```

You can run it in the background on HPC using:

```
nohup python main.py --conf configs/config.yaml > output.log &
```
Use tb flag to enable tensorboard
```
python main.py --conf configs/config.yaml -tb 
```
use tbpath `-tbpth ./logs` for custom log path

2. Configure tensorboard to view relevant experiment parameters TODO.

3. Collect experiment results

| Impl | Encoder | Arch | Upsampling | K | a1 | a2 | a3 | abs_rel | log_rms | rms | sq_rel | Link |
|---------------|--------------|-----------------|--------|--------|--------|---------|---------|-------|--------|--------|--------| -------- |
| Paper[2] | resnet50 | UNet | bilinear | &#x2717; | 0.8777 | 0.959 | 0.981 | 0.115  | 0.193 | 4.863 | 0.903 | - |
| CamLess[5] | resneXt50 | UNet | ESPCN | &#10003; | 0.891 | 0.964 | 0.983 | 0.106  | 0.182  | 4.482 | 0.750 | - |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.8784 | 0.9654 | 0.9867 | 0.109 | 0.1887 | 4.327 | 0.661 | [Link](https://storage.googleapis.com/depthestimation-weights/baseline-resnet-unet.zip) |
| Ours | resnet50 | UNet++ | bilinear | 0.8808 | 0.9607 | 0.9835  | 0.1483 | 0.2372 | 6.000 | 3.709 | [Link](https://storage.googleapis.com/depthestimation-weights/resnet-unetplusplus.zip) |
| Ours | convnext-tiny | UNet | bilinear | &#x2717; | **0.9145** | 0.9682 | 0.9852  | **0.09386** | 0.1776 | 3.953 | **0.5298** | [Link](https://storage.googleapis.com/depthestimation-weights/convnext-unet.zip) |
| Ours | convnext-tiny | UNet | ESPCN | &#x2717; | 0.8384 | 0.961 | 0.989  | 0.1224 | 0.1892 | **3.886** | 0.587 | [Link](https://storage.googleapis.com/depthestimation-weights/convnext-unet-espcn.zip) |
| Ours | convnext-tiny | UNet++ | ESPCN | &#x2717; | 0.8229 | **0.9751** | **0.9902**  | 0.1234 | 0.1933 | 4.07 | 0.6039 | [Link](https://storage.googleapis.com/depthestimation-weights/convnext-unetplusplus-espcn.zip) |
| Ours | resnet50 | UNet | bilinear | &#10003; | 0.8752 | 0.9575 | 0.9814  | 0.1125 | 0.1984 | 4.55 | 0.6957 | [Link](https://storage.googleapis.com/depthestimation-weights/resnet-unet-camnet.zip) |
| Ours | convnext-tiny | UNet | bilinear | &#10003; | 0.7346 | 0.8911 | 0.9491  | 0.1828 | 0.2981 | 7.515 | 1.474 | [Link](https://storage.googleapis.com/depthestimation-weights/convnext-unet-camnet.zip) |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.9111 | 0.9733 | 0.9878  | 0.1005 | **0.1693** | 3.978 | 0.5615 | [Link](https://storage.googleapis.com/depthestimation-weights/resnet-unet-espcn.zip) |

Baseline Model (Monodepth2)|  ConvNext + UNet Implementation
:-------------------------:|:-------------------------:
![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/fig6.png)  |  ![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/WSP-2UP4_pred_convnext-unet_espcn-False.jpg)

|Sample Video | Monodepth2 Output |
|-------------|-------------------|
|![testVideo](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo.gif)|![baseline](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-baseline-resnet-unet.gif)|

|ConvNeXt-UNet Output | ConvNeXt-UNet++-ESPCN Output |
|---------------------|----------------------------|
|![convnext-unet](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-convnext-unet.gif)|![convnext-unetplusplus-espcn](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-convnext-unetplusplus-espcn.gif)|

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contributors

* Mayank Poddar
* Akash Mishra
* Shikhar Vaish
