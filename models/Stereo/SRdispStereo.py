import utils.experiment
import utils.data
from utils import myUtils
from .SRStereo import SRStereo
from utils.warp import warpAndCat


class SRdispStereo(SRStereo):
    # imgL: RGB value range 0~1
    # output: RGB value range 0~1
    # mask: useless in this case
    def predict(self, batch: utils.data.Batch, mask=(1, 1)):
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        batch.lowestResRGBs(cated)
        outputs = super().predict(batch=batch, mask=mask)
        utils.data.packWarpTo(warpTos=warpTos, outputs=outputs)
        return outputs

    def train(self, batch: utils.data.Batch, progress=0):
        batch.assertScales(2) # TODO: Added support for one scale input
        cated, warpTos = warpAndCat(batch.lastScaleBatch())
        batch = batch.detach()
        batch.lowestResRGBs(cated)
        losses, outputs = super().train(batch=batch, progress=progress)
        utils.data.packWarpTo(warpTos=warpTos, outputs=outputs)
        return losses, outputs


