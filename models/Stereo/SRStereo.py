import utils.experiment
import torch.optim as optim
import torch
import torch.nn as nn
import utils.data
import utils.imProcess
from utils import myUtils
from .Stereo import Stereo
from .. import SR
from apex import amp


class RawSRStereo(nn.Module):
    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__()
        self.sr = myUtils.getNNmoduleFromModel(sr)
        self.stereo = myUtils.getNNmoduleFromModel(stereo)
        self.updateSR = True

    def forward(self, left, right):
        with torch.set_grad_enabled(self.updateSR and self.training):
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
        self.outMaxDisp = stereo.outMaxDisp
        self.sr = sr
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
        self.model.module.updateSR = lossWeights[0] >= 0
        if lossWeights[0] < 0:
            lossWeights[0] = 0
        super().setLossWeights(lossWeights)
        self.stereo.setLossWeights(lossWeights[1:])
        self.sr.setLossWeights(lossWeights[0])

    def initModel(self):
        self.model = RawSRStereo(self.sr, self.stereo)
        self.getParamNum()

    def packOutputs(self, outputs, imgs: utils.imProcess.Imgs = None):
        return self.stereo.packOutputs(outputs, self.sr.packOutputs(outputs, imgs))

    def loss(self, output: utils.imProcess.Imgs, gt: tuple):
        gtSrs, dispHigh, dispLow = gt
        loss = utils.data.NameValues()
        hasSr = not dispHigh is dispLow
        if all([img is not None for img in gtSrs]) and self.sr.lossWeights > 0:
            outputSrL = output['outputSrL']
            outputSrR = output['outputSrR']
            if not hasSr:
                outputSrL = nn.AvgPool2d((2, 2))(outputSrL)
                outputSrR = nn.AvgPool2d((2, 2))(outputSrR)
            # average lossSrL/R
            lossSr = (self.sr.loss(output={'outputSr': outputSrL}, gt=gtSrs[0])
                + self.sr.loss(output={'outputSr': outputSrR}, gt=gtSrs[1])) / 2
            loss.add(lossSr)

        if not all([disp is None for disp in (dispHigh, dispLow)]):
            loss.add(self.stereo.loss(output=output, gt=(dispHigh, dispLow)))

        return loss

    def train(self, batch: utils.data.Batch, progress=0):

        return self.trainBothSides(
            batch.lowestResRGBs(),
            list(zip([batch.highestResRGBs(),
                      batch.highestResRGBs()[::-1] if batch.highestResDisps()[1] is not None else None
                      ],
                     batch.highestResDisps(),
                     batch.lowestResDisps()))
        )

    def predict(self, batch: utils.data.Batch, mask=(1, 1)):
        outputs = self.sr.predict(batch=batch.lastScaleBatch())
        batch = batch.detach()
        batch.lowestResRGBs((outputs['outputSrL'], outputs['outputSrR']))
        outputs.update(self.stereo.predict(batch=batch, mask=mask))
        return outputs

    def test(self, batch: utils.data.Batch, evalType: str):
        loss, outputs = super().test(batch=batch, evalType=evalType)
        if len(batch) == 8:
            loss.update(self.sr.testOutput(outputs=outputs, gt=batch.highResRGBs(), evalType=evalType))
        return loss, outputs

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
