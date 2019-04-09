from utils.warp import warpAndCat
from .EDSR import *


class EDSRdisp(EDSR):
    def initModel(self):
        self.model = RawEDSR(cInput=6)

    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        batch.lowestResRGBs(cated)
        outputs = super().predict(batch, mask)
        for warpTo, side in zip(warpTos, ('L', 'R')):
            outputs.addImg(name='warpTo' + side, img=warpTo)

        return outputs

    def train(self, batch: myUtils.Batch):
        batch.assertScales(2)
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        losses, outputs = self.trainBothSides(cated, batch.highResRGBs())
        for warpTo, side in zip(warpTos, ('L', 'R')):
            outputs.addImg(name='warpTo' + side, img=warpTo)
        return losses, outputs


