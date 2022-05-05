# Depth Estimation using self supervised learning (Monocular images)


* [Environment Setup and Model Training](#env)
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

<a name="ref"></a>
<li>[1] Godard, Cl ́ement, et al., ”Digging into self-supervised monocular depth estimation.” Proceedings of the
IEEE/CVF International Conference on Computer Vision. 2019. arXiv:1806.01260
<li>[2] Source Code of Monodepth2: GitHub - nianticlabs/monodepth2: [ICCV 2019] Monocular depth estimation
from a single image.
<li>[3] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang, “UNet++: A Nested
U-Net Architecture for Medical Image Segmentation”. arXiv:1807.10165
<li>[4] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, “A Con-
vNet for the 2020s”. arXiv:2201.03545.
