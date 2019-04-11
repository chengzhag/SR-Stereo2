from utils import myUtils
from .SRStereo import SRStereo
from utils.warp import warpAndCat


class SRdispStereo(SRStereo):
    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    # mask: useless in this case
    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        batch.lowestResRGBs(cated)
        outputs = super().predict(batch=batch, mask=mask)
        myUtils.packWarpTo(warpTos=warpTos, outputs=outputs)
        return outputs

    def train(self, batch: myUtils.Batch, progress=0):
        batch.assertScales(2)
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        losses, outputs = self.trainBothSides(
            cated,
            list(zip([batch.highResRGBs(), ] * 2, batch.highResDisps(), batch.lowResDisps()))
        )
        myUtils.packWarpTo(warpTos=warpTos, outputs=outputs)
        return losses, outputs


