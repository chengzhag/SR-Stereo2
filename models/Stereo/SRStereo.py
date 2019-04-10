import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .Stereo import Stereo
from .. import SR
from evaluation import evalFcn
from utils.warp import warpAndCat


class RawSRStereo(nn.Module):
    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__()

        def getModel(model):
            model = model.model
            if hasattr(model, 'module'):
                model = model.module
            return model

        self.sr = getModel(sr)
        self.stereo = getModel(stereo)

    def forward(self, left, right, updateSR=True):
        with torch.no_grad() if updateSR else None:
            outputSrL = self.sr.forward(left)['outputSr']
            outputSrR = self.sr.forward(right)['outputSr']

        output = self.stereo.forward(outputSrL, outputSrR)
        output['outputSrL'] = outputSrL
        output['outputSrR'] = outputSrR
        return output


class SRStereo(Stereo):

    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__(
            maxDisp=stereo.maxDisp, dispScale=stereo.dispScale, cuda=stereo.cuda, half=stereo.half)
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

    def initModel(self):
        self.model = RawSRStereo(self.sr, self.stereo)

    def loss(self, output: myUtils.Imgs, gt: tuple, kitti=False):
        gtSrs, dispHigh, dispLow = gt
        loss = myUtils.NameValues()
        if all([img is not None for img in gtSrs]):
            # average lossSrL/R
            lossSr = self.sr.loss(output={'outputSr': output['outputSrL']}, gt=gtSrs[0]) \
                .add(self.sr.loss(output={'outputSr': output['outputSrR']}, gt=gtSrs[1])) \
                .div(2)
            loss.update(lossSr)

        if not all([disp is None for disp in (dispHigh, dispLow)]):
            loss.add(self.stereo.loss(output=output, gt=(dispHigh, dispLow), kitti=kitti))

        return loss

    def trainOneSide(self, input, gt, kitti=False):
        self.model.train()
        self.optimizer.zero_grad()
        rawOutputs = self.model.forward(*input, updateSR=self.lossWeights[0] >= 0)
        output = self.stereo.packOutputs(rawOutputs, self.sr.packOutputs(rawOutputs))
        loss = self.loss(output=output, gt=gt, kitti=kitti)
        with self.ampHandle.scale_loss(loss['loss'], self.optimizer) as scaledLoss:
            scaledLoss.backward()
        self.optimizer.step()

        output.addImg(name='outputDisp', img=output['outputDisp'][2].detach(), range=self.outMaxDisp)
        return loss.dataItem(), output

    def trainBothSides(self, inputs, gts, kitti=False):
        losses = myUtils.NameValues()
        outputs = myUtils.Imgs()
        for input, gt, process, side in zip(
                (inputs, inputs[::-1]), gts,
                (lambda im: im, myUtils.flipLR),
                ('L', 'R')
        ):
            if not all([g is None for g in gt]):
                loss, output = self.trainOneSide(
                    *process([input, gt]),
                    kitti=kitti
                )
                losses.update(nameValues=loss, suffix=side)
                outputs.update(imgs=process(output), suffix=side)

    def train(self, batch: myUtils.Batch, kitti=False, progress=0):
        batch.assertScales(2)

        return self.trainBothSides(
            batch.lowestResRGBs(),
            list(zip([batch.highResRGBs(), ] * 2, batch.highResDisps(), batch.lowResDisps())),
            kitti=kitti
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

        if type(checkpointDir) in (list, tuple) and len(checkpointDir) == 2:
            # Load pretrained SR and Stereo weights
            self.sr.load(checkpointDir[0])
            self.stereo.load(checkpointDir[1])
            return None, None
        elif type(checkpointDir) is str:
            # Load fintuned SRStereo weights
            return super(SRStereo, self).load(checkpointDir)
        else:
            raise Exception('Error: SRStereo need 2 checkpoints SR/Stereo or 1 checkpoint SRStereo to load!')
