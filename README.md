# Depth Estimation using self supervised learning (Monocular images)


* [Environment Setup](#env)



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