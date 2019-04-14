import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .SRdispStereo import SRdispStereo
from .Stereo import Stereo
from .. import SR
import torch.nn.functional as F
from evaluation import evalFcn
import random


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
        outputs = myUtils.Imgs()
        with torch.no_grad():
            outSRs = [F.interpolate(
                lowResInput, scale_factor=2, mode='bilinear', align_corners=False
            ) for lowResInput in batch.lowestResRGBs()]
            initialBatch = myUtils.Batch(4, cuda=batch.cuda, half=batch.half)
            initialBatch.lowestResRGBs(outSRs)
            outputs.update(self.stereo.predict(initialBatch), suffix='_0')
            for outSr, side in zip(outSRs, ('L', 'R')):
                outputs.addImg('outputSr' + side + '_0', outSr)
            for it in range(1, itRefine + 1):
                batch.lowestResDisps(outputs.getImgPair('outputDisp', suffix=('_%d' % (it - 1))))
                itOutputs = super().predict(batch.detach())
                outputs.update(itOutputs, suffix=('_%d' % it))
                if it == itRefine:
                    outputs.update(itOutputs)

        return outputs

    def test(self, batch: myUtils.Batch, evalType: str):
        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        mask = [disp is not None for disp in disps]
        outputs = self.predict(batch, mask)

        def getIt(it: int):
            output = myUtils.Imgs()
            for key in outputs.keys():
                if key.endswith('_%d' % it):
                    output[key[:key.find('_')]] = outputs[key]
            return output

        loss = myUtils.NameValues()
        for it in range(self.itRefine + 1):
            output = getIt(it)
            lossIt = self.testOutput(outputs=output, gt=disps, evalType=evalType)
            if len(batch) == 8:
                lossIt.update(self.sr.testOutput(outputs=output, gt=batch.highResRGBs(), evalType=evalType))
            loss.update(lossIt, '_%d' % it)
            if it == self.itRefine:
                loss.update(lossIt)

        return loss, outputs






