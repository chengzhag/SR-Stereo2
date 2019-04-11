import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
from .Stereo import Stereo
from .. import SR


class RawSRStereo(nn.Module):
    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__()
        self.sr = myUtils.getNNmoduleFromModel(sr)
        self.stereo = myUtils.getNNmoduleFromModel(stereo)
        self.updateSR = True

    def forward(self, left, right):
        with torch.no_grad() if self.updateSR else None:
            outputSrL = self.sr.forward(left)['outputSr']
            outputSrR = self.sr.forward(right)['outputSr']

        output = self.stereo.forward(outputSrL, outputSrR)
        output['outputSrL'] = outputSrL
        output['outputSrR'] = outputSrR
        return output


class SRStereo(Stereo):
    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__(
            kitti=stereo.kitti, maxDisp=stereo.maxDisp, dispScale=stereo.dispScale, cuda=stereo.cuda, half=stereo.half)
        stereo.optimizer = None
        sr.optimizer = None
        self.stereo = stereo
        self.sr = sr
        self.initModel()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001, betas=(0.9, 0.999)
        )
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def setLossWeights(self, lossWeights):
        super().setLossWeights(lossWeights)
        self.stereo.setLossWeights(lossWeights[1:])
        self.sr.setLossWeights(lossWeights[0])
        self.model.module.updateSR = lossWeights[0] >= 0

    def initModel(self):
        self.model = RawSRStereo(self.sr, self.stereo)

    def packOutputs(self, outputs, imgs: myUtils.Imgs = None):
        return self.stereo.packOutputs(outputs, self.sr.packOutputs(outputs, imgs))

    def loss(self, output: myUtils.Imgs, gt: tuple):
        gtSrs, dispHigh, dispLow = gt
        loss = myUtils.NameValues()
        if all([img is not None for img in gtSrs]):
            # average lossSrL/R
            lossSr = self.sr.loss(output={'outputSr': output['outputSrL']}, gt=gtSrs[0]) \
                .add(self.sr.loss(output={'outputSr': output['outputSrR']}, gt=gtSrs[1])) \
                .div(2)
            loss.update(lossSr)

        if not all([disp is None for disp in (dispHigh, dispLow)]):
            loss.add(self.stereo.loss(output=output, gt=(dispHigh, dispLow)))

        return loss

    def train(self, batch: myUtils.Batch, progress=0):
        batch.assertScales(2)
        return self.trainBothSides(
            batch.lowestResRGBs(),
            list(zip([batch.highResRGBs(), ] * 2, batch.highResDisps(), batch.lowResDisps()))
        )

    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        outputs = self.sr.predict(batch=batch)
        batch.lowestResRGBs((outputs['outputSrL'], outputs['outputSrR']))
        outputs.update(self.stereo.predict(batch=batch, mask=mask))
        return outputs

    def load(self, checkpointDir):
        if checkpointDir is None:
            return None, None

        if type(checkpointDir) in (list, tuple):
            if len(checkpointDir) == 2:
                self.sr.load(checkpointDir[0])
                self.stereo.load(checkpointDir[1])
                return None, None
            elif len(checkpointDir) == 1:
                return super().load(checkpointDir)
        elif type(checkpointDir) is str:
            return super().load(checkpointDir)
        raise Exception('Error: SRStereo need 2 checkpoints SR/Stereo or 1 checkpoint SRStereo to load!')
