import utils.experiment
import torch.optim as optim
import torch
import torch.nn as nn
import utils.data
import utils.imProcess
from utils import myUtils
from .Stereo import Stereo
from .Feature import Feature
from apex import amp


class RawFeatureStereo(nn.Module):
    def __init__(self, feature: Feature, stereoBody: Stereo):
        super().__init__()
        self.feature = myUtils.getNNmoduleFromModel(feature)
        self.stereoBody = myUtils.getNNmoduleFromModel(stereoBody)
        self.updateFeature = True

    def forward(self, left, right):
        with torch.set_grad_enabled(self.updateFeature and self.training):
            outputFeatureL = self.feature.forward(left)['outputFeature']
            outputFeatureR = self.feature.forward(right)['outputFeature']

        output = self.stereoBody.forward(outputFeatureL, outputFeatureR)
        return output


class FeatureStereo(Stereo):
    def __init__(self, feature: Feature, stereoBody: Stereo):
        super().__init__(
            kitti=stereoBody.kitti,
            maxDisp=stereoBody.maxDisp,
            dispScale=stereoBody.dispScale,
            cuda=stereoBody.cuda,
            half=stereoBody.half)
        stereoBody.optimizer = None
        feature.optimizer = None
        self.stereoBody = stereoBody
        self.outMaxDisp = stereoBody.outMaxDisp
        self.feature = feature
        self.initModel()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001, betas=(0.9, 0.999)
        )
        if self.cuda:
            self.model.cuda()
            self.model, self.optimizer = amp.initialize(models=self.model, optimizers=self.optimizer, enabled=self.half)
            self.model = nn.DataParallel(self.model)

    def setLossWeights(self, lossWeights):
        self.model.module.updateFeature = lossWeights[0] > 0
        print('UpdateFeature ' + ('Enabled' if self.model.module.updateFeature else 'Disabled'))
        if lossWeights[0] < 0:
            lossWeights[0] = 0
        super().setLossWeights(lossWeights)
        self.feature.setLossWeights(lossWeights[0])
        self.stereoBody.setLossWeights(lossWeights[1])

    def initModel(self):
        self.model = RawFeatureStereo(self.feature, self.stereoBody)
        self.getParamNum()

    def packOutputs(self, outputs, imgs: utils.imProcess.Imgs = None):
        return self.stereoBody.packOutputs(outputs, self.feature.packOutputs(outputs, imgs))

    def loss(self, output: utils.imProcess.Imgs, gt: tuple):
        return self.stereoBody.loss(output=output, gt=gt)

    def train(self, batch: utils.data.Batch, progress=0):
        return self.trainBothSides(batch.lowestResRGBs(), batch.lowestResDisps())

    def load(self, checkpointDir):
        if checkpointDir is None:
            return None, None

        if type(checkpointDir) in (list, tuple):
            if len(checkpointDir) == 2:
                self.feature.load(checkpointDir[0], strict=False)
                self.stereoBody.load(checkpointDir[1], strict=False)
                return None, None
            elif len(checkpointDir) == 1:
                return super().load(checkpointDir, strict=True)
        elif type(checkpointDir) is str:
            return super().load(checkpointDir)
        raise Exception('Error: SRStereo need 2 checkpoints SR/Stereo or 1 checkpoint SRStereo to load!')
