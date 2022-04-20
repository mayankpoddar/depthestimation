import random
import PIL.Image as pil

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T

from utils import pilLoader

class CustomDataset(data.Dataset):
    def __init__(self, dataPath, filenames, height, width, frameIdxs, numScales, train=False):
        super(CustomDataset, self).__init__()
        self.dataPath = dataPath
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frameIdxs = frameIdxs
        self.numScales = numScales
        self.train = train
        self.interpolation = T.InterpolationMode.LANCZOS
        self.loader = pilLoader
        self.toTensor = T.ToTensor()
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.resize = {}
        for scaleNum in range(self.numScales):
            scale = 2**scaleNum
            self.resize[scaleNum] = T.Resize((self.height//scale, self.width//scale),
                                             interpolation=self.interpolation)
        self.loadDepth = self.checkDepth()

    def preprocess(self, inputs, colorAugmentations):
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.numScales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.toTensor(frame)
                inputs[(n + "_aug", im, i)] = self.toTensor(colorAugmentations(frame))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        colorAugmentationsFlag = self.train and random.random() > 0.5
        flipFlag = self.train and random.random() > 0.5
        line = self.filenames[index].split()
        directory = line[0]
        frameIdx = 0
        side = None
        if len(line) == 3:
            frameIdx = int(line[1])
            side = line[2]
        for fi in self.frameIdxs:
            if fi == "s":
                otherSide = {"r": "l", "l": "r"}[side]
                inputs[("color", fi, -1)] = self.getColor(directory, frameIdx, otherSide, flipFlag)
            else:
                inputs[("color", fi, -1)] = self.getColor(directory, frameIdx + fi, side, flipFlag)
        for scale in range(self.numScales):
            K = self.K.copy()
            K[0, :] *= self.width//(2**scale)
            K[1, :] *= self.height//(2**scale)
            inverseK = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inverseK)
        colorAugmentations = (lambda x: x)
        if colorAugmentationsFlag:
            colorAugmentations = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.preprocess(inputs, colorAugmentations)
        for fi in self.frameIdxs:
            del inputs[("color", fi, -1)]
            del inputs[("color_aug", fi, -1)]
        if self.loadDepth:
            depthGroundTruth = self.getDepth(directory, frameIdx, side, flipFlag)
            inputs["depth_gt"] = np.expand_dims(depthGroundTruth, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        if "s" in self.frameIdxs:
            stereoT = np.eye(4, dtype=np.float32)
            baselineSign = -1 if flipFlag else 1
            sideSign = -1 if side == "l" else 1
            stereoT[0, 3] = sideSign * baselineSign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereoT)
        return inputs

    def checkDepth(self):
        raise NotImplementedError

    def getColor(self, directory, frameIdx, otherSide, flip):
        raise NotImplementedError

    def getDepth(self, directory, frameIdx, otherSide, flip):
        raise NotImplementedError
