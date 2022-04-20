import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from Models.ESPCN import ESPCN

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation=True):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(int(inChannels), int(outChannels), kernel_size=3)
        self.act = nn.ELU()

    def forward(self, x):
        if self.activation:
            out = self.act(self.conv(self.pad(x)))
        else:
            out = self.conv(self.pad(x))
        return out

class DepthDecoderModel(nn.Module):
    def __init__(self, numChannelsEncoder, espcn=False):
        super(DepthDecoderModel, self).__init__()
        self.numChannelsEncoder = numChannelsEncoder
        self.espcn = espcn
        self.numChannelsDecoder = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for layer in range(4, -1, -1):
            inChannels = self.numChannelsEncoder[-1] if layer == 4 else self.numChannelsDecoder[layer+1]
            outChannels = self.numChannelsDecoder[layer]
            self.convs[("upconv", layer, 0)] = ConvBlock(inChannels, outChannels, activation=True)
            inChannels = self.numChannelsDecoder[layer]
            if self.espcn:
                self.convs[("espcn", layer)] = ESPCN(2, outChannels, inChannels, activation=False)
            if layer > 0:
                inChannels += self.numChannelsEncoder[layer-1]
            outChannels = self.numChannelsDecoder[layer]
            self.convs[("upconv", layer, 1)] = ConvBlock(inChannels, outChannels, activation=True)
        for scale in range(4):
            self.convs[("dispconv", scale)] = ConvBlock(self.numChannelsDecoder[scale], 1, activation=False)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, inputFeatures):
        self.outputs = {}
        x = inputFeatures[-1]
        for layer in range(4, -1, -1):
            x = self.convs[("upconv", layer, 0)](x)
            if self.espcn:
                x = [self.convs[("espcn", layer)](x)]
            else:
                x = [nn.functional.interpolate(x, scale_factor=sf, mode="nearest")]
            if layer > 0:
                x += [inputFeatures[layer-1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", layer, 1)](x)
            if layer < 4:
                out = self.convs[("dispconv", layer)](x)
                self.outputs[("disp", layer)] = torch.sigmoid(out)
        return self.outputs
    
class PoseDecoderModel(nn.Module):
    def __init__(self, numChannelsEncoder, numFeatures=1, numFrames=2):
        super(PoseDecoderModel, self).__init__()
        self.numChannelsEncoder = numChannelsEncoder
        self.numFeaturesInput = numFeatures
        self.numFramesPredict = numFrames
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.numChannelsEncoder[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(self.numFeaturesInput*256, 256, 3, 1, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6*self.numFramesPredict, 1)
        self.relu = nn.ReLU()
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, inputFeatures):
        lastFeatures = [feature[-1] for feature in inputFeatures]
        catFeatures = [self.relu(self.convs["squeeze"](feature)) for feature in lastFeatures]
        catFeatures = torch.cat(catFeatures, 1)
        out = catFeatures
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.numFramesPredict, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation
