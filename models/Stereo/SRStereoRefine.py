import torch.optim as optim
import torch
import torch.nn as nn

import utils.data
import utils.experiment
import utils.imProcess
from utils import myUtils
import collections
from .SRdispStereo import SRdispStereo
from .Stereo import Stereo
from .. import SR
import torch.nn.functional as F
from evaluation import evalFcn
import random
from utils.warp import warpAndCat


class SRStereoRefine(SRdispStereo):
    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__(sr=sr, stereo=stereo)
        self.itRefine=0

    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    # mask: useless in this case
    def predict(self, batch, mask=(1,1), itRefine=None):
        if itRefine is None:
            itRefine = self.itRefine

        # initialize SR output from low res input
        outputs = utils.data.Imgs()
        with torch.no_grad():
            outSRs = [F.interpolate(
                lowResInput, scale_factor=2, mode='bilinear', align_corners=False
            ) for lowResInput in batch.lowestResRGBs()]
            initialBatch = utils.data.Batch(4, cuda=batch.cuda, half=batch.half)
            initialBatch.lowestResRGBs(outSRs)
            itOutputs = self.stereo.predict(initialBatch)
            outputs.update(itOutputs, suffix='_0')
            if itRefine == 0:
                outputs.update(itOutputs)
            for outSr, side in zip(outSRs, ('L', 'R')):
                outputs.addImg('outputSr' + side + '_0', outSr)
            for it in range(1, itRefine + 1):
                batch.lowestResDisps(outputs.getImgPair('outputDisp', suffix=('_%d' % (it - 1))))
                itOutputs = super().predict(batch.detach(), mask=mask if it == itRefine else (1, 1))
                outputs.update(itOutputs, suffix=('_%d' % it))
                if it == itRefine:
                    outputs.update(itOutputs)

        return outputs

    def test(self, batch: utils.data.Batch, evalType: str):
        disps = batch.lowestResDisps()
        utils.imProcess.assertDisp(*disps)

        mask = [disp is not None for disp in disps]
        outputs = self.predict(batch, mask)

        loss = utils.data.NameValues()
        for it in range(self.itRefine + 1):
            output = outputs.getIt(it)
            lossIt = self.testOutput(outputs=output, gt=disps, evalType=evalType)
            if len(batch) == 8:
                lossIt.update(self.sr.testOutput(outputs=output, gt=batch.highResRGBs(), evalType=evalType))
            loss.update(lossIt, '_%d' % it)
            if it == self.itRefine:
                loss.update(lossIt)

        return loss, outputs

    # weights: weights of
    #   SR output losses (lossSR),
    #   SR disparity map losses (lossDispHigh),
    #   normal sized disparity map losses (lossDispLow)
    def train(self, batch: utils.data.Batch, progress=0):
        # probability of training with dispsOut as input:
        # progress = [0, 1]: p = [0, 1]
        if random.random() < progress or self.kitti:
            itRefine = random.randint(0, 2)
            dispChoice = itRefine
            outputs = self.predict(batch.lastScaleBatch(), itRefine=itRefine)
            warpBatch = utils.data.Batch(batch.lowestResRGBs() + [outputs['outputDispL'], outputs['outputDispR']],
                                         cuda=batch.cuda, half=batch.half)
        else:
            warpBatch = batch.lastScaleBatch()
            dispChoice = -1

        cated, warpTos = warpAndCat(warpBatch)
        batch = batch.detach()
        batch.lowestResRGBs(cated)
        losses, outputs = super(SRdispStereo, self).train(batch=batch, progress=progress)
        utils.data.packWarpTo(warpTos=warpTos, outputs=outputs)
        losses['dispChoice'] = dispChoice

        return losses, outputs


