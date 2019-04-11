from utils.warp import warpAndCat
from .EDSR import *


class SRdisp(SR):
    def __init__(self, sr: SR):
        super().__init__(cuda=sr.cuda, half=sr.half)
        self.sr = sr
        self.optimizer = sr.optimizer
        self.model = sr.model

    def setLossWeights(self, lossWeights):
        super().setLossWeights(lossWeights)
        self.sr.setLossWeights(lossWeights)

    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        batch.lowestResRGBs(cated)
        outputs = self.sr.predict(batch, mask)
        myUtils.packWarpTo(warpTos=warpTos, outputs=outputs)
        return outputs

    def train(self, batch: myUtils.Batch):
        batch.assertScales(2)
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        losses, outputs = self.sr.trainBothSides(cated, batch.highResRGBs())
        myUtils.packWarpTo(warpTos=warpTos, outputs=outputs)
        return losses, outputs


